from dataclasses import dataclass

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask

from .joints import ARM_JOINT_NAMES, BASE_THETA, BASE_X, BASE_Y
from .model import ARM_JOINTS, BASE_JOINTS, MERGED_ARM, build_model

PUBLIC_NAMES = {
    MERGED_ARM: "arm_joint",
    BASE_JOINTS["x"]: BASE_X,
    BASE_JOINTS["y"]: BASE_Y,
    BASE_JOINTS["theta"]: BASE_THETA,
}
assert [PUBLIC_NAMES.get(n, n) for n in ARM_JOINTS] == ARM_JOINT_NAMES


@dataclass
class IKResult:
    joints: dict
    converged: bool
    position_error: float  # m
    orientation_error: float  # rad, about the constrained axes
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
        self._name_to_public = {n: PUBLIC_NAMES.get(n, n) for n in self.model.names[1:]}
        self._public_to_name = {v: k for k, v in self._name_to_public.items()}

        cost = np.full(self.model.nv, posture_cost)
        for public, c in (posture_costs or {}).items():
            if public in self._public_to_name:
                cost[
                    self.model.idx_vs[
                        self.model.getJointId(self._public_to_name[public])
                    ]
                ] = c
        self._posture_task = PostureTask(cost=cost)

    @property
    def joint_names(self):
        return list(self._name_to_public.values())

    def to_q(self, joints: dict) -> np.ndarray:
        q = pin.neutral(self.model)
        for name, public in self._name_to_public.items():
            if public in joints:
                q[self.model.idx_qs[self.model.getJointId(name)]] = joints[public]
            elif not name.startswith("base_"):
                raise KeyError(f"missing joint {public}")
        return q

    def to_joints(self, q: np.ndarray) -> dict:
        return {
            public: float(q[self.model.idx_qs[self.model.getJointId(name)]])
            for name, public in self._name_to_public.items()
        }

    def _base_to_world(self, q_start: np.ndarray) -> pin.SE3:
        q0 = q_start.copy()
        for name in BASE_JOINTS.values():
            if self.model.existJointName(name):
                q0[self.model.idx_qs[self.model.getJointId(name)]] = 0.0
        pin.framesForwardKinematics(self.model, self.data, q0)
        return self.data.oMf[self.model.getFrameId(self.base_frame)].copy()

    def fk(self, joints: dict) -> pin.SE3:
        """ee pose in base_link at the start. Base joints are the base travel."""
        q = self.to_q(joints)
        base = self._base_to_world(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return base.actInv(self.data.oMf[self.model.getFrameId(self.ee_frame)])

    def base_motion(self, joints: dict) -> pin.SE3:
        """base_link after the base travel, in base_link at the start."""
        q = self.to_q(joints)
        start = self._base_to_world(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return start.actInv(self.data.oMf[self.model.getFrameId(self.base_frame)])

    def solve(self, start: dict, target: pin.SE3, posture: dict = None) -> IKResult:
        """`target`: ee pose in base_link at the start. `posture` only selects among
        equal solutions."""
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
        err = self._frame_task.compute_error(config)
        return float(np.linalg.norm(err[:3])), float(
            np.linalg.norm(err[3:][self._rot_axes])
        )
