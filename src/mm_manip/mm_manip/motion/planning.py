from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from mm_ik.solver import ARM_JOINT_NAMES, IKSolver

from ..errors import SkillError

# No cost about z: the wrist yaw can aim, and the gripper stays level.
FREE_YAW_ORIENTATION_COST = (0.5, 0.5, 0.0)

# The base translates. It turns and moves the arm less.
BASE_STAGE_POSTURE_COST = 1e-2
BASE_STAGE_POSTURE_COSTS = {"base_x": 1e-4, "base_y": 1e-4, "base_theta": 1e-1}

MAX_AIM_YAW = np.deg2rad(10)
MAX_BASE_ROTATION = np.deg2rad(45)


class PlanningError(SkillError):
    pass


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


def _arm_only(joints: dict) -> dict:
    return {k: joints[k] for k in ARM_JOINT_NAMES}


def _acceptable(result, accept_position_error=0.0, accept_orientation_error=0.0):
    return result.converged or (
        result.position_error <= accept_position_error
        and result.orientation_error <= accept_orientation_error
    )


def _check(result, what, accept_position_error=0.0, accept_orientation_error=0.0):
    if _acceptable(result, accept_position_error, accept_orientation_error):
        return
    raise PlanningError(
        f"IK did not converge for {what}: position error {result.position_error * 100:.1f} cm, "
        f"orientation error {np.rad2deg(result.orientation_error):.1f} deg"
    )


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


def reach(
    arm_solver,
    start: dict,
    target,
    base_solver=None,
    base_posture=None,
    accept_position_error=0.0,
    accept_orientation_error=0.0,
):
    """Returns (base motion, arm joints). The base moves only if necessary."""
    accept = {
        "accept_position_error": accept_position_error,
        "accept_orientation_error": accept_orientation_error,
    }
    result = arm_solver.solve(start, target)
    if _acceptable(result, **accept):
        return {}, _arm_only(result.joints)
    if base_solver is None:
        _check(result, "target", **accept)
    placed = base_solver.solve(start, target, posture=base_posture)
    _check(placed, "target (with base)", **accept)
    moved = base_solver.base_motion(placed.joints)
    result = arm_solver.solve(_arm_only(placed.joints), moved.actInv(target))
    _check(result, "target (after base motion)", **accept)
    return _base_only(placed.joints), _arm_only(result.joints)


def _base_only(joints: dict) -> dict:
    return {k: v for k, v in joints.items() if k.startswith("base_")}


def _plan(solver, arm_solver, start, poses, posture, accept) -> GraspPlan:
    pre = solver.solve(start, poses["pregrasp"], posture=posture)
    _check(pre, "pre-grasp", **accept)
    base = _base_only(pre.joints)
    moved = solver.base_motion(pre.joints)

    targets = [moved.actInv(p) for p in poses["approach"][1:] + [poses["lift"]]]
    arm = []
    joints = _arm_only(pre.joints)
    for i, target in enumerate(targets):
        r = arm_solver.solve(joints, target)
        _check(
            r,
            f"approach waypoint {i + 1}" if i < len(targets) - 1 else "lift",
            **accept,
        )
        joints = _arm_only(r.joints)
        arm.append(joints)

    return GraspPlan(base, _arm_only(pre.joints), arm[:-1], arm[-1], poses)


def solve_waypoints(
    solver,
    start: dict,
    targets,
    what: str = "waypoint",
    accept_position_error=0.0,
    accept_orientation_error=0.0,
    postures=None,
    partial: bool = False,
) -> list:
    """With partial, stop at the first target that fails, and return the path to it."""
    accept = {
        "accept_position_error": accept_position_error,
        "accept_orientation_error": accept_orientation_error,
    }
    postures = postures or [None] * len(targets)
    joints, path = start, []
    for i, (target, posture) in enumerate(zip(targets, postures, strict=True)):
        result = solver.solve(joints, target, posture=posture)
        if partial and not _acceptable(result, **accept):
            break
        _check(result, f"{what} {i}", **accept)
        joints = result.joints
        path.append(joints)
    return path
