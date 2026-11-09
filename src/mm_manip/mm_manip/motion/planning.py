import numpy as np

from mm_ik.joints import arm_only, base_only

from ..errors import SkillError


class PlanningError(SkillError):
    pass


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


def reach_path(
    arm_solver,
    start: dict,
    targets: list,
    names: list,
    base_solver=None,
    base_posture=None,
    **accept,
):
    """Returns (base motion, arm joints per target). The base moves once, first,
    and only if the arm alone cannot reach every target."""
    try:
        return {}, solve_waypoints(arm_solver, start, targets, names=names, **accept)
    except PlanningError as arm_error:
        if base_solver is None:
            raise
        try:
            placed = base_solver.solve(start, targets[0], posture=base_posture)
            check(placed, f"{names[0]} (with base)", **accept)
            moved = base_solver.base_motion(placed.joints)
            path = solve_waypoints(
                arm_solver,
                arm_only(placed.joints),
                [moved.actInv(target) for target in targets],
                names=names,
                **accept,
            )
        except PlanningError as base_error:
            raise PlanningError(
                f"{base_error} (arm alone: {arm_error})"
            ) from base_error
        return base_only(placed.joints), path


def reach(
    arm_solver, start: dict, target, base_solver=None, base_posture=None, **accept
):
    """Returns (base motion, arm joints). The base moves only if necessary."""
    base, path = reach_path(
        arm_solver, start, [target], ["target"], base_solver, base_posture, **accept
    )
    return base, path[0]


def solve_waypoints(
    solver,
    start: dict,
    targets,
    names=None,
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
    names = names or [f"waypoint {i}" for i in range(len(targets))]
    postures = postures or [None] * len(targets)
    joints, path = start, []
    for target, name, posture in zip(targets, names, postures, strict=True):
        result = solver.solve(joints, target, posture=posture)
        if partial and not acceptable(result, **accept):
            break
        check(result, name, **accept)
        joints = result.joints
        path.append(joints)
    return path
