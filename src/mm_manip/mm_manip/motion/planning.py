"""Grasp planning: target poses -> joint waypoints."""

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from mm_ik.solver import ARM_JOINT_NAMES, IKSolver

# IK orientation cost per grasp_center_link axis (x, y, z). Zero on z lets the wrist
# yaw aim at objects slightly off the gripper's line while the gripper stays level.
FREE_YAW_ORIENTATION_COST = (0.5, 0.5, 0.0)


# Base stage of the IK (only used when the arm alone cannot reach): moving the base
# is cheap, rotating it less so (a holonomic base should translate), and the arm is
# dearer and biased to a comfortable posture, so the base does the travelling and the
# arm keeps room for the approach.
BASE_STAGE_POSTURE_COST = 1e-2
BASE_STAGE_POSTURE_COSTS = {"base_x": 1e-4, "base_y": 1e-4, "base_theta": 1e-1}

# Defaults for make_solvers.
MAX_AIM_YAW = np.deg2rad(10)
MAX_BASE_ROTATION = np.deg2rad(45)


class PlanningError(RuntimeError):
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
    """(arm_solver, base_solver) for plan_grasp; base_solver is None when base_dofs is empty.

    With free_yaw, the wrist yaw may aim at most max_aim_yaw away from grasp_yaw
    (about +-5 cm sideways at 10 deg); anything further is left to the base.
    """
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
    """The gripper points along +x of grasp_center_link."""
    return rotation[:, 0]


def grasp_poses(
    center, rotation, pregrasp_distance=0.10, n_waypoints=6, lift_height=0.10
):
    """Build the pose sequence for one grasp.

    Args:
        center: (3,) object center in base_link.
        rotation: (3, 3) fixed gripper orientation in base_link.
        pregrasp_distance: back-off along the approach axis.
        n_waypoints: poses on the straight line pre-grasp -> center, both ends included.
        lift_height: how far to lift along base_link +z after closing.

    Returns:
        dict with 'pregrasp' (SE3), 'approach' (list of SE3, starts at pregrasp,
        ends at the object), 'lift' (SE3).
    """
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
    """Gripper orientation in base_link for the given wrist angles.

    Only the wrist joints affect the orientation (lift and arm are prismatic), so
    the other joints are filled with zeros.
    """
    joints = {name: 0.0 for name in ARM_JOINT_NAMES}
    joints.update(wrist)
    return solver.fk(joints).rotation


@dataclass
class GraspPlan:
    base: dict  # base_x/base_y/base_theta to execute first, {} when the base is locked
    pregrasp: dict  # arm joints
    approach: list  # arm joints, pre-grasp -> object (pre-grasp excluded)
    lift: dict  # arm joints
    poses: dict = field(
        default_factory=dict
    )  # from grasp_poses(), in the starting base_link


def _arm_only(joints: dict) -> dict:
    return {k: joints[k] for k in ARM_JOINT_NAMES}


def _acceptable(result, accept_position_error=0.0, accept_orientation_error=0.0):
    return result.converged or (
        result.position_error <= accept_position_error
        and result.orientation_error <= accept_orientation_error
    )


def _check(result, what, accept_position_error=0.0, accept_orientation_error=0.0):
    """Accept converged results, and unconverged ones whose residual is small enough.

    With the base locked the arm cannot move sideways, so a target slightly off the
    gripper's line never converges; a ~1 cm miss is still fine for grasping.
    """
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
    """Plan base motion + arm waypoints for one grasp, using the base only if needed.

    First the arm alone (base locked) tries the whole grasp. If any pose is out of
    reach and a base_solver is given, the base_solver places the pre-grasp with the
    arm biased to base_posture, the base moves there once, and the arm alone solves
    the approach line and the lift as seen from the moved base.

    Raises PlanningError if any pose is unreachable, i.e. its IK neither converges
    nor ends within accept_position_error (m) and accept_orientation_error (rad).
    """
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
    """(base motion, arm joints) to put the ee at `target`, using the base only if needed."""
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
        r = arm_solver.solve(joints, target)  # warm start from the previous point
        _check(
            r,
            f"approach waypoint {i + 1}" if i < len(targets) - 1 else "lift",
            **accept,
        )
        joints = _arm_only(r.joints)
        arm.append(joints)

    return GraspPlan(base, _arm_only(pre.joints), arm[:-1], arm[-1], poses)
