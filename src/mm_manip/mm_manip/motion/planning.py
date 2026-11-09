import numpy as np

from mm_ik.solver import ARM_JOINT_NAMES

from ..errors import SkillError


class PlanningError(SkillError):
    pass


def arm_only(joints: dict) -> dict:
    return {k: joints[k] for k in ARM_JOINT_NAMES}


def acceptable(result, accept_position_error=0.0, accept_orientation_error=0.0):
    return result.converged or (
        result.position_error <= accept_position_error
        and result.orientation_error <= accept_orientation_error
    )


def check(result, what, accept_position_error=0.0, accept_orientation_error=0.0):
    if acceptable(result, accept_position_error, accept_orientation_error):
        return
    raise PlanningError(
        f"IK did not converge for {what}: position error {result.position_error * 100:.1f} cm, "
        f"orientation error {np.rad2deg(result.orientation_error):.1f} deg"
    )


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
    if acceptable(result, **accept):
        return {}, arm_only(result.joints)
    if base_solver is None:
        check(result, "target", **accept)
    placed = base_solver.solve(start, target, posture=base_posture)
    check(placed, "target (with base)", **accept)
    moved = base_solver.base_motion(placed.joints)
    result = arm_solver.solve(arm_only(placed.joints), moved.actInv(target))
    check(result, "target (after base motion)", **accept)
    return base_only(placed.joints), arm_only(result.joints)


def base_only(joints: dict) -> dict:
    return {k: v for k, v in joints.items() if k.startswith("base_")}


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
        if partial and not acceptable(result, **accept):
            break
        check(result, f"{what} {i}", **accept)
        joints = result.joints
        path.append(joints)
    return path
