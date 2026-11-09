from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from mm_ik.solver import IKSolver

from ...errors import ArticulateFailed
from ...motion.planning import arm_only
from ...motion.velocity import BASE_JOINTS, BASE_THETA, BASE_X, BASE_Y, TrackingGains
from .door import (
    Door,
    DoorType,
    HandleType,
    door_from_points,
    hinge_axis,
    horizontal,
    into_handle,
    opening_arc,
    radius,
    with_handle_offset,
)
from .plan import plan_arc

DEFAULT_OPEN_ANGLE = np.pi / 2


# The gripper closes along the y axis of grasp_center_link. A horizontal bar
# needs it vertical: -pi/2 (the roll range is about -4.3..1.1 rad).
HANDLE_ROLL = {
    HandleType.BAR_VERTICAL: 0.0,
    HandleType.KNOB: 0.0,
    HandleType.BAR_HORIZONTAL: -np.pi / 2,
}


class Phase(IntEnum):
    # Values match the feedback phases in Articulate.action.
    MOVING_TO_START = 0
    DETECTING = 1
    APPROACHING = 2
    GRASPING = 3
    OPENING = 4


@dataclass(frozen=True)
class ArticulateGoal:
    prompt: str = ""
    handle: np.ndarray | None = None  # base frame
    hinge: np.ndarray | None = None  # base frame, on the vertical hinge axis
    open_angle: float = 0.0  # rad, 0 = DEFAULT_OPEN_ANGLE


@dataclass(frozen=True)
class Outcome:
    door: Door
    opened: float  # rad
    asked: float  # rad


def run(goal: ArticulateGoal, cfg, shared, ctx) -> Outcome:
    if ctx.robot.urdf is None:
        raise ArticulateFailed("no /robot_description received yet")
    gains = TrackingGains(
        max_velocity=cfg.path_max_velocity,
        tolerance=cfg.path_tolerance,
        gain=cfg.tracking_gain,
        min_segment=cfg.path_min_segment,
    )

    ctx.phase(Phase.MOVING_TO_START)
    if shared.dry_run:
        current = cfg.start_pose
    else:
        ctx.robot.follow_path([cfg.start_pose], ctx.cancel, gains)
        ctx.cancel.sleep(shared.settle_time)
        current = ctx.robot.arm_joints()

    ctx.phase(Phase.DETECTING)
    if goal.handle is not None:
        door = door_from_points(goal.handle, goal.hinge)
    else:
        door = ctx.locate(goal.prompt, ctx.cancel)
        if door.handle_from_depth:
            door, protrusion, moved = into_handle(door, cfg.handle_grasp_depth)
            ctx.log.info(
                f"handle front {protrusion * 100:.1f} cm out of the door; "
                f"grasping {moved * 100:.1f} cm behind it"
            )
        else:
            ctx.log.warn(
                "no handle in depth in front of the door: using the door plane "
                f"+ handle_offset {cfg.handle_offset} m"
            )
            door = with_handle_offset(door, cfg.handle_offset)
        # HACK: the grasps land left of and above the handle.
        door = replace(door, handle=door.handle + [0.0, -0.015, -0.05])
    ctx.report.keep(door=door)
    ctx.report.show(door.handle)
    ctx.log.info(
        f"door {door.type.name}, handle {door.handle_type.name} "
        f"at {door.handle.round(3)} in the base frame"
    )
    if door.type not in (DoorType.REVOLUTE_LEFT, DoorType.REVOLUTE_RIGHT):
        raise NotImplementedError(f"opening a {door.type.name.lower()} door")
    if door.handle_type not in HANDLE_ROLL:
        raise NotImplementedError(f"a {door.handle_type.name.lower()} handle")

    asked = goal.open_angle or DEFAULT_OPEN_ANGLE
    arc = opening_arc(door, asked, cfg.n_waypoints, HANDLE_ROLL[door.handle_type])
    ctx.report.show(arc.axis, arc.poses)
    ctx.log.info(f"hinge axis at {arc.axis.round(3)}, {len(arc.poses)} arc poses")
    plan = plan_arc(arc, current, cfg, shared, ctx.robot.urdf)
    end = plan.path[-1]
    ctx.log.info(
        f"arc: radius {radius(door, arc) * 100:.1f} cm, opens "
        f"{np.rad2deg(plan.opened):.0f} of {np.rad2deg(asked):.0f} deg "
        f"({plan.strategy}); base ends at ({end[BASE_X]:.2f}, {end[BASE_Y]:.2f}) m, "
        f"{np.rad2deg(end[BASE_THETA]):.0f} deg"
    )
    approach = {j: round(plan.grasp[j], 3) for j in BASE_JOINTS if j in plan.grasp}
    if approach:
        ctx.log.info(f"base moves to reach the handle: {approach}")
    outcome = Outcome(door, plan.opened, asked)
    if shared.dry_run:
        ctx.report.ghost([arm_only(plan.grasp)] + [arm_only(p) for p in plan.path])
        return outcome

    ctx.phase(Phase.APPROACHING)
    if shared.use_gripper:
        ctx.robot.open_gripper(ctx.cancel)
    else:
        ctx.log.warn("use_gripper is false: not opening the gripper")
    ctx.robot.follow_path([plan.grasp], ctx.cancel, gains)
    _log_grasp_error(ctx, door, arc, plan, shared)

    ctx.phase(Phase.GRASPING)
    if shared.use_gripper:
        if not ctx.robot.close_gripper(ctx.cancel):
            raise ArticulateFailed("gripper closed all the way: missed the handle")
    else:
        ctx.log.warn("use_gripper is false: not closing the gripper")

    ctx.phase(Phase.OPENING)
    ctx.robot.follow_path(plan.path, ctx.cancel, gains)
    return outcome


def _log_grasp_error(ctx, door, arc, plan, shared):
    """Debug only. Log the gripper error at the grasp: along the door, out, up."""
    try:
        fk = IKSolver(
            ctx.robot.urdf,
            ("x", "y", "theta"),
            ee_frame=shared.ee_frame,
            base_frame=shared.base_frame,
        ).fk
        base = {j: plan.grasp.get(j, 0.0) for j in BASE_JOINTS}
        measured_joints = ctx.robot.arm_joints()
        target = arc.poses[0].translation
        planned = fk({**arm_only(plan.grasp), **base}).translation
        measured = fk({**measured_joints, **base}).translation
    except Exception as e:  # noqa: BLE001  debugging must not fail the goal
        ctx.log.warn(f"grasp error check: {e}")
        return
    out = horizontal(door.normal)
    along = horizontal(hinge_axis(door) - door.handle)

    def split(p):
        e = p - target
        return f"along {e @ along:+.3f}, out {e @ out:+.3f}, up {e[2]:+.3f}"

    joints = ", ".join(
        f"{j.removesuffix('_joint')} {plan.grasp[j]:.3f}/{measured_joints[j]:.3f}"
        for j in arm_only(plan.grasp)
    )
    ctx.log.info(
        f"grasp error (m, gripper - handle): planned {split(planned)}; "
        f"measured {split(measured)}; joints planned/measured: {joints}"
    )
