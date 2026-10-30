"""Offline differential IK with pink.

Targets are poses of `ee_frame` expressed in `base_link` at the moment of solving.
The planar base always starts at the origin, so the returned base values are the
relative motion (dx, dy, dtheta) the base must make before the arm moves.
"""

from dataclasses import dataclass

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask

from .model import ARM_JOINTS, BASE_JOINT, MERGED_ARM, build_model

# Names used outside this package.
# The merged arm is `arm_joint`.
PUBLIC_NAMES = {MERGED_ARM: "arm_joint"}
ARM_JOINT_NAMES = [PUBLIC_NAMES.get(n, n) for n in ARM_JOINTS]
BASE_NAMES = ["base_x", "base_y", "base_theta"]
# Planar base at the start of motion: (x, y, cos(theta), sin(theta)).
_BASE_ZERO = np.array([0.0, 0.0, 1.0, 0.0])


@dataclass
class IKResult:
    joints: dict  # public joint name -> position
    converged: bool
    position_error: float  # m
    orientation_error: float  # rad
    iterations: int


class IKSolver:
    def __init__(
        self,
        urdf_xml: str,
        base_dofs=(),
        ee_frame: str = "grasp_center_link",
        base_frame: str = "base_link",
        joint_limits=None,
        position_cost: float = 1.0,
        orientation_cost=0.5,
        posture_cost: float = 1e-3,
        posture_costs=None,
        dt: float = 0.1,
        damping: float = 1e-6,
        max_iterations: int = 300,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-2,
        settle_tolerance: float = 1e-4,
        qp_solver: str = "daqp",
    ):
        self.model = build_model(urdf_xml, base_dofs, joint_limits)
        self.data = self.model.createData()
        for frame in (ee_frame, base_frame):
            if not self.model.existFrame(frame):
                raise ValueError(f"frame {frame} not in URDF")
        self.ee_frame = ee_frame
        self.base_frame = base_frame
        self.dt = dt
        self.damping = damping
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.settle_tolerance = settle_tolerance
        self.qp_solver = qp_solver

        costs = np.broadcast_to(np.asarray(orientation_cost, dtype=float), (3,))
        self._rot_axes = costs > 0
        self._frame_task = FrameTask(
            ee_frame, position_cost=position_cost, orientation_cost=costs.copy()
        )
        # Arm joints are 1-dof: one q and one v index each. The planar base takes
        # q[i:i+4] = (x, y, cos, sin) and v[j:j+3] = (vx, vy, wz).
        self._arm_q = {}
        self._v_index = {}
        for name in ARM_JOINTS:
            jid = self.model.getJointId(name)
            self._arm_q[PUBLIC_NAMES.get(name, name)] = self.model.idx_qs[jid]
            self._v_index[PUBLIC_NAMES.get(name, name)] = self.model.idx_vs[jid]
        self._base_q = None
        if self.model.existJointName(BASE_JOINT):
            jid = self.model.getJointId(BASE_JOINT)
            self._base_q = self.model.idx_qs[jid]
            for k, public in enumerate(BASE_NAMES):
                self._v_index[public] = self.model.idx_vs[jid] + k

        cost = np.full(self.model.nv, posture_cost)
        for public, c in (posture_costs or {}).items():
            if public in self._v_index:
                cost[self._v_index[public]] = c
        self._posture_task = PostureTask(cost=cost)

    @property
    def joint_names(self):
        """Public names of the free joints."""
        return list(self._v_index)

    def to_q(self, joints: dict) -> np.ndarray:
        """Joint dict -> configuration. Base joints default to 0; arm joints are required."""
        q = pin.neutral(self.model)
        for public, i in self._arm_q.items():
            if public not in joints:
                raise KeyError(f"missing joint {public}")
            q[i] = joints[public]
        if self._base_q is not None:
            x, y, theta = (joints.get(public, 0.0) for public in BASE_NAMES)
            q[self._base_q : self._base_q + 4] = x, y, np.cos(theta), np.sin(theta)
        return q

    def to_joints(self, q: np.ndarray) -> dict:
        joints = {public: float(q[i]) for public, i in self._arm_q.items()}
        if self._base_q is not None:
            x, y, c, s = q[self._base_q : self._base_q + 4]
            joints.update(
                base_x=float(x), base_y=float(y), base_theta=float(np.arctan2(s, c))
            )
        return joints

    def _base_to_world(self, q_start: np.ndarray) -> pin.SE3:
        """Pose of base_link in the IK world, with base joints at 0 (start of motion)."""
        q0 = q_start.copy()
        if self._base_q is not None:
            q0[self._base_q : self._base_q + 4] = _BASE_ZERO
        pin.framesForwardKinematics(self.model, self.data, q0)
        return self.data.oMf[self.model.getFrameId(self.base_frame)].copy()

    def fk(self, joints: dict) -> pin.SE3:
        """Pose of ee_frame in base_link at the *start* of motion (base joints act as offsets)."""
        q = self.to_q(joints)
        base = self._base_to_world(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return base.actInv(self.data.oMf[self.model.getFrameId(self.ee_frame)])

    def base_motion(self, joints: dict) -> pin.SE3:
        """Pose of base_link after the base moves by joints' base_x/y/theta, in the starting base_link."""
        q = self.to_q(joints)
        start = self._base_to_world(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return start.actInv(self.data.oMf[self.model.getFrameId(self.base_frame)])

    def solve(self, start: dict, target: pin.SE3, posture: dict = None) -> IKResult:
        """Solve for `target` (ee pose in base_link at the start of motion).

        `start` may carry base_x/base_y/base_theta (e.g. from a previous solve);
        missing base joints default to 0.
        `posture` biases the listed joints toward these values instead of `start`
        (e.g. a comfortable arm extension); it only picks among redundant solutions.
        """
        q = self.to_q(start)
        q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        self._frame_task.set_target(self._base_to_world(q) * target)
        self._posture_task.set_target(
            self.to_q({**self.to_joints(q), **(posture or {})})
        )
        tasks = [self._frame_task, self._posture_task]
        config = pink.Configuration(self.model, self.data, q)

        for i in range(1, self.max_iterations + 1):
            v = pink.solve_ik(
                config,
                tasks,
                self.dt,
                solver=self.qp_solver,
                damping=self.damping,
                safety_break=False,
            )
            reached = self._reached(config)
            if reached and np.linalg.norm(v) * self.dt < self.settle_tolerance:
                return IKResult(
                    self.to_joints(config.q), True, *self._errors(config), i - 1
                )
            config.integrate_inplace(v, self.dt)

        pos_err, rot_err = self._errors(config)
        return IKResult(
            self.to_joints(config.q),
            self._reached(config),
            pos_err,
            rot_err,
            self.max_iterations,
        )

    def _reached(self, config):
        pos_err, rot_err = self._errors(config)
        return (
            pos_err < self.position_tolerance and rot_err < self.orientation_tolerance
        )

    def _errors(self, config):
        """Position error (m) and orientation error (rad) about the constrained axes only."""
        err = self._frame_task.compute_error(config)
        return float(np.linalg.norm(err[:3])), float(
            np.linalg.norm(err[3:][self._rot_axes])
        )

    def solve_path(self, start: dict, targets) -> list:
        """Solve a sequence of targets, warm-starting each from the previous solution."""
        results = []
        joints = start
        for target in targets:
            result = self.solve(joints, target)
            results.append(result)
            if not result.converged:
                break
            joints = result.joints
        return results
