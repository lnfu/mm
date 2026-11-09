from dataclasses import dataclass, field, replace
from enum import IntEnum

import numpy as np
import pinocchio as pin

from mm_ik.solver import ARM_JOINT_NAMES, IKSolver

from ..errors import SkillError
from ..motion.planning import PlanningError, reach, solve_waypoints
from ..motion.velocity import BASE_JOINTS, BASE_THETA, BASE_X, BASE_Y, TrackingGains
from ..perception.door import Door, DoorType, HandleType
from .primitives import OutOfReach

DEFAULT_OPEN_ANGLE = np.pi / 2
Z = np.array([0.0, 0.0, 1.0])

# Posture costs. The arm joints have 1e-3.
BASE_TRANSLATION_COST = 0.1
FREE_TURN_COST = 1e-3
PRESCRIBED_TURN_COST = 1.0
APPROACH_ARM_POSTURE_COST = 1e-2
APPROACH_TRANSLATION_COST = 1e-4

# The gripper closes along the y axis of grasp_center_link. A horizontal bar
# needs it vertical: -pi/2 (the roll range is about -4.3..1.1 rad).
HANDLE_ROLL = {
    HandleType.BAR_VERTICAL: 0.0,
    HandleType.KNOB: 0.0,
    HandleType.BAR_HORIZONTAL: -np.pi / 2,
}


class ArticulateFailed(SkillError):
    pass


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
class ArcPlan:
    grasp: dict  # arm joints and the base motion to the grasp
    path: list  # base joints are relative to the base at the grasp
    opened: float  # rad
    strategy: str
    base_travel: float = field(default=0.0)


@dataclass(frozen=True)
class Outcome:
    door: Door
    opened: float  # rad
    asked: float  # rad


@dataclass(frozen=True)
class Arc:
    axis: np.ndarray
    poses: list  # poses[0] grasps the handle
    angles: np.ndarray  # rad
    sign: float  # +1: the door turns counterclockwise, seen from above


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
        door = ctx.locator.locate(goal.prompt, ctx.cancel)
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
    ctx.observer.door(door)
    if door.type not in (DoorType.REVOLUTE_LEFT, DoorType.REVOLUTE_RIGHT):
        raise NotImplementedError(f"opening a {door.type.name.lower()} door")
    if door.handle_type not in HANDLE_ROLL:
        raise NotImplementedError(f"a {door.handle_type.name.lower()} handle")

    asked = goal.open_angle or DEFAULT_OPEN_ANGLE
    arc = opening_arc(door, asked, cfg.n_waypoints, HANDLE_ROLL[door.handle_type])
    ctx.observer.arc(arc)
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
        ctx.observer.ghost([_arm(plan.grasp)] + [_arm(p) for p in plan.path])
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
        planned = fk({**_arm(plan.grasp), **base}).translation
        measured = fk({**measured_joints, **base}).translation
    except Exception as e:  # noqa: BLE001  debugging must not fail the goal
        ctx.log.warn(f"grasp error check: {e}")
        return
    out = _horizontal(door.normal)
    along = _horizontal(hinge_axis(door) - door.handle)

    def split(p):
        e = p - target
        return f"along {e @ along:+.3f}, out {e @ out:+.3f}, up {e[2]:+.3f}"

    joints = ", ".join(
        f"{j.removesuffix('_joint')} {plan.grasp[j]:.3f}/{measured_joints[j]:.3f}"
        for j in _arm(plan.grasp)
    )
    ctx.log.info(
        f"grasp error (m, gripper - handle): planned {split(planned)}; "
        f"measured {split(measured)}; joints planned/measured: {joints}"
    )


def door_from_points(handle, hinge) -> Door:
    handle, hinge = np.asarray(handle, float), np.asarray(hinge, float)
    along = _horizontal(hinge - handle)
    normal = np.cross(Z, along)
    if normal @ -_horizontal(handle) < 0:
        normal = -normal
    door_type = (
        DoorType.REVOLUTE_LEFT if hinge[1] > handle[1] else DoorType.REVOLUTE_RIGHT
    )
    hinge_at_handle = np.array([hinge[0], hinge[1], handle[2]])
    corners = {"hinge": hinge_at_handle}
    return Door(door_type, HandleType.BAR_VERTICAL, handle, corners, normal)


def into_handle(door: Door, depth: float, clearance: float = 0.005):
    """Move the handle `depth` into the door. Returns (door, protrusion, moved) in m."""
    out = _horizontal(door.normal)
    protrusion = float((door.handle - door.corners["top_left"]) @ door.normal)
    moved = min(depth, max(protrusion - clearance, 0.0))
    return replace(door, handle=door.handle - moved * out), protrusion, moved


def with_handle_offset(door: Door, offset: float) -> Door:
    return replace(door, handle=door.handle + offset * _horizontal(door.normal))


def hinge_axis(door: Door) -> np.ndarray:
    if "hinge" in door.corners:
        point = door.corners["hinge"]
    else:
        side = "left" if door.type == DoorType.REVOLUTE_LEFT else "right"
        top, bottom = door.corners[f"top_{side}"], door.corners[f"bottom_{side}"]
        edge = bottom - top
        length = np.linalg.norm(edge)
        if length < 1e-6:
            raise ArticulateFailed(f"the {side} door corners coincide")
        edge /= length
        point = top + ((door.handle - top) @ edge) * edge
    return np.array([point[0], point[1], door.handle[2]])


def radius(door: Door, arc: Arc) -> float:
    return float(np.linalg.norm((door.handle - arc.axis)[:2]))


def opening_arc(door: Door, angle: float, n_waypoints: int, roll: float = 0.0) -> Arc:
    if n_waypoints < 2:
        raise ValueError("n_waypoints must be >= 2")
    axis = hinge_axis(door)
    arm = door.handle - axis
    if np.linalg.norm(arm[:2]) < 0.02:
        raise ArticulateFailed("the handle is on the hinge axis")
    # Turn so that the handle first moves toward the robot.
    sign = 1.0 if np.cross(Z, arm) @ door.normal > 0 else -1.0
    pointing = -_horizontal(door.normal)
    start = _rot_z(np.arctan2(pointing[1], pointing[0])) @ _rot_x(roll)
    angles = np.linspace(0.0, angle, n_waypoints)
    poses = []
    for theta in angles:
        turn = _rot_z(sign * theta)
        poses.append(pin.SE3(turn @ start, axis + turn @ arm))
    return Arc(axis, poses, angles, sign)


def plan_arc(arc: Arc, current: dict, cfg, shared, urdf) -> ArcPlan:
    accept = {
        "accept_position_error": cfg.ik_accept_position_error,
        "accept_orientation_error": np.deg2rad(cfg.ik_accept_orientation_error_deg),
    }
    frames = {"ee_frame": shared.ee_frame, "base_frame": shared.base_frame}
    travel, turn = cfg.max_base_translation, np.deg2rad(cfg.max_base_rotation_deg)
    limits = {
        "base_x_joint": (-travel, travel),
        "base_y_joint": (-travel, travel),
        "base_theta_joint": (-turn, turn),
    }

    def base_solver(turn_cost, translation_cost=BASE_TRANSLATION_COST, **kwargs):
        return IKSolver(
            urdf,
            ("x", "y", "theta"),
            joint_limits=limits,
            posture_costs={
                BASE_X: translation_cost,
                BASE_Y: translation_cost,
                BASE_THETA: turn_cost,
            },
            **frames,
            **kwargs,
        )

    try:
        arm_solver = IKSolver(urdf, (), **frames)
        free, prescribed = (
            base_solver(FREE_TURN_COST),
            base_solver(PRESCRIBED_TURN_COST),
        )
        approach = base_solver(
            PRESCRIBED_TURN_COST,
            APPROACH_TRANSLATION_COST,
            posture_cost=APPROACH_ARM_POSTURE_COST,
        )
    except ValueError as e:
        raise ArticulateFailed(f"cannot build IK model: {e}") from e

    try:
        base, grasp = reach(
            arm_solver,
            current,
            arc.poses[0],
            base_solver=approach,
            base_posture={"arm_joint": cfg.approach_arm, BASE_THETA: 0.0},
            **accept,
        )
    except PlanningError as e:
        raise OutOfReach(f"handle out of reach: {e}") from e
    moved = approach.base_motion({**grasp, **base})
    start = {**grasp, **dict.fromkeys(BASE_JOINTS, 0.0)}
    targets = [moved.actInv(pose) for pose in arc.poses[1:]]
    angles = arc.angles[1:]

    # Keep the plan that opens most, then moves the base least.
    candidates = [("base turn free", free, None)] + [
        (
            f"base turn {ratio:g} x door",
            prescribed,
            [{BASE_THETA: arc.sign * ratio * a} for a in angles],
        )
        for ratio in cfg.base_turn_ratios
    ]
    plans = []
    for name, solver, postures in candidates:
        path = solve_waypoints(
            solver, start, targets, postures=postures, partial=True, **accept
        )
        if path:
            travel = max(max(abs(p[BASE_X]), abs(p[BASE_Y])) for p in path)
            plans.append((len(path), -travel, name, path, travel))
    if not plans:
        raise PlanningError("no strategy opens the door past the grasp")
    n, _, name, path, travel = max(plans, key=lambda plan: plan[:2])
    return ArcPlan(
        {**grasp, **base},
        [{**_arm(p), **{j: p[j] for j in BASE_JOINTS}} for p in path],
        float(angles[n - 1]),
        name,
        travel,
    )


def _arm(joints: dict) -> dict:
    return {k: joints[k] for k in ARM_JOINT_NAMES}


def _horizontal(v) -> np.ndarray:
    v = np.array([v[0], v[1], 0.0])
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ArticulateFailed("a door direction is vertical")
    return v / norm


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
