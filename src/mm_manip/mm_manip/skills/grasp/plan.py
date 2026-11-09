from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from mm_ik.solver import ARM_JOINT_NAMES, IKSolver

from ...motion.planning import PlanningError, arm_only, base_only, check

# No cost about z: the wrist yaw can aim, and the gripper stays level.
FREE_YAW_ORIENTATION_COST = (0.5, 0.5, 0.0)


# The base translates. It turns and moves the arm less.
BASE_STAGE_POSTURE_COST = 1e-2


BASE_STAGE_POSTURE_COSTS = {"base_x": 1e-4, "base_y": 1e-4, "base_theta": 1e-1}


MAX_AIM_YAW = np.deg2rad(10)


MAX_BASE_ROTATION = np.deg2rad(45)


def make_solvers(
    urdf_xml,
    base_dofs=(),
    free_yaw=True,
    grasp_yaw=0.0,
    max_aim_yaw=MAX_AIM_YAW,
    max_base_translation=0.6,
    max_base_rotation=MAX_BASE_ROTATION,
    **kwargs,
):
    limits = {
        "base_x_joint": (-max_base_translation, max_base_translation),
        "base_y_joint": (-max_base_translation, max_base_translation),
        "base_theta_joint": (-max_base_rotation, max_base_rotation),
    }
    if free_yaw:
        kwargs["orientation_cost"] = FREE_YAW_ORIENTATION_COST
        limits["wrist_yaw_joint"] = (grasp_yaw - max_aim_yaw, grasp_yaw + max_aim_yaw)
    arm_limits = {k: v for k, v in limits.items() if not k.startswith("base_")}
    arm_solver = IKSolver(urdf_xml, (), joint_limits=arm_limits, **kwargs)
    if not base_dofs:
        return arm_solver, None
    base_solver = IKSolver(
        urdf_xml,
        base_dofs,
        joint_limits=limits,
        posture_cost=BASE_STAGE_POSTURE_COST,
        posture_costs=BASE_STAGE_POSTURE_COSTS,
        **kwargs,
    )
    return arm_solver, base_solver


def approach_axis(rotation: np.ndarray) -> np.ndarray:
    return rotation[:, 0]


def grasp_poses(
    center, rotation, pregrasp_distance=0.10, n_waypoints=6, lift_height=0.10
):
    if n_waypoints < 2:
        raise ValueError("n_waypoints must be >= 2")
    center = np.asarray(center, float)
    pre = center - pregrasp_distance * approach_axis(rotation)
    approach = [
        pin.SE3(rotation, pre + s * (center - pre))
        for s in np.linspace(0.0, 1.0, n_waypoints)
    ]
    lift = pin.SE3(rotation, center + np.array([0.0, 0.0, lift_height]))
    return {"pregrasp": approach[0], "approach": approach, "lift": lift}


def fixed_grasp_rotation(solver, wrist: dict) -> np.ndarray:
    joints = {name: 0.0 for name in ARM_JOINT_NAMES}
    joints.update(wrist)
    return solver.fk(joints).rotation


@dataclass
class GraspPlan:
    base: dict  # base motion to do first; {} if the base stays
    pregrasp: dict
    approach: list  # without the pre-grasp
    lift: dict
    poses: dict = field(default_factory=dict)  # in the start base_link


def plan_grasp(
    arm_solver,
    start: dict,
    center,
    rotation,
    pregrasp_distance=0.10,
    n_waypoints=6,
    lift_height=0.10,
    accept_position_error=0.0,
    accept_orientation_error=0.0,
    base_solver=None,
    base_posture=None,
) -> GraspPlan:
    """Try the arm alone. If it cannot reach, move the base once to the pre-grasp."""
    accept = {
        "accept_position_error": accept_position_error,
        "accept_orientation_error": accept_orientation_error,
    }
    poses = grasp_poses(center, rotation, pregrasp_distance, n_waypoints, lift_height)
    try:
        return _plan(arm_solver, arm_solver, start, poses, None, accept)
    except PlanningError as arm_error:
        if base_solver is None:
            raise
        try:
            return _plan(base_solver, arm_solver, start, poses, base_posture, accept)
        except PlanningError as base_error:
            raise PlanningError(
                f"{base_error} (arm alone: {arm_error})"
            ) from base_error


def _plan(solver, arm_solver, start, poses, posture, accept) -> GraspPlan:
    pre = solver.solve(start, poses["pregrasp"], posture=posture)
    check(pre, "pre-grasp", **accept)
    base = base_only(pre.joints)
    moved = solver.base_motion(pre.joints)

    targets = [moved.actInv(p) for p in poses["approach"][1:] + [poses["lift"]]]
    arm = []
    joints = arm_only(pre.joints)
    for i, target in enumerate(targets):
        r = arm_solver.solve(joints, target)
        check(
            r,
            f"approach waypoint {i + 1}" if i < len(targets) - 1 else "lift",
            **accept,
        )
        joints = arm_only(r.joints)
        arm.append(joints)

    return GraspPlan(base, arm_only(pre.joints), arm[:-1], arm[-1], poses)
