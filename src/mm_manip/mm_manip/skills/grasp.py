"""Grasp skill."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pinocchio as pin

from ..motion.base_control import BaseGains
from ..motion.planning import fixed_grasp_rotation, make_solvers, plan_grasp
from ..perception.localize import center_and_width
from .primitives import approach_with_base


class GraspFailed(RuntimeError):
    pass


class Phase(IntEnum):
    """Same values as the phases in mm_interfaces/action/Grasp feedback."""

    MOVING_TO_START = 0
    DETECTING = 1
    APPROACHING = 2
    GRASPING = 3
    LIFTING = 4


@dataclass(frozen=True)
class GraspGoal:
    prompt: str = ""  # detect this with the camera, or
    point: np.ndarray | None = None  # grasp here (base frame), without the camera


def run(goal: GraspGoal, cfg, ctx) -> np.ndarray:
    """Grasp once; returns the object center in the final base frame.

    ctx.observer gets phase(Phase), target(center), plan(center, plan) and, in
    dry_run, ghost(arm waypoints) instead of any motion.
    Raises GraspFailed, OutOfReach, PlanningError, Canceled, and the robot's and
    the locator's errors.
    """
    arm_solver, base_solver = _solvers(cfg, ctx.robot.urdf)
    rotation = fixed_grasp_rotation(arm_solver, cfg.grasp_wrist)

    ctx.phase(Phase.MOVING_TO_START)
    if cfg.dry_run:
        current = cfg.start_pose
    else:
        ctx.robot.move_arm(ctx.robot.arm_joints(), [cfg.start_pose], ctx.should_cancel)
        ctx.sleep(cfg.settle_time)
        current = ctx.robot.arm_joints()

    ctx.phase(Phase.DETECTING)
    center = goal.point if goal.point is not None else _detect(goal, cfg, ctx, rotation)
    ctx.observer.target(center)

    def replan(current, center):
        plan = plan_grasp(
            arm_solver,
            current,
            center,
            rotation,
            pregrasp_distance=cfg.pregrasp_distance,
            n_waypoints=cfg.n_waypoints,
            lift_height=cfg.lift_height,
            accept_position_error=cfg.ik_accept_position_error,
            accept_orientation_error=np.deg2rad(cfg.ik_accept_orientation_error_deg),
            base_solver=base_solver,
            base_posture={
                "arm_joint": cfg.base_stage_arm,
                "wrist_yaw_joint": cfg.grasp_wrist["wrist_yaw_joint"],
            },
        )
        ctx.observer.plan(center, plan)
        return plan

    def relocate(plan, center):
        ctx.phase(Phase.DETECTING)
        if goal.point is None:
            center = _detect(goal, cfg, ctx, rotation)
        else:  # no camera: trust the planned base motion
            moved = base_solver.base_motion({**plan.pregrasp, **plan.base})
            center = moved.actInv(pin.SE3(np.eye(3), center)).translation
        ctx.observer.target(center)
        ctx.phase(Phase.APPROACHING)
        return center

    plan = replan(current, center)
    if cfg.dry_run:
        ctx.observer.ghost([plan.pregrasp] + plan.approach + [plan.lift])
        return center

    ctx.phase(Phase.APPROACHING)
    plan, current, center = approach_with_base(
        ctx,
        plan,
        replan,
        relocate,
        current,
        center,
        max_moves=cfg.max_base_moves,
        gains=BaseGains(
            max_linear=cfg.base_max_linear,
            max_angular=cfg.base_max_angular,
            linear_gain=cfg.base_linear_gain,
            angular_gain=cfg.base_angular_gain,
        ),
        settle_time=cfg.settle_time,
    )

    if cfg.use_gripper:
        ctx.robot.open_gripper(ctx.should_cancel)
    else:
        ctx.log.warn("use_gripper is false: not opening the gripper")
    ctx.robot.move_arm(current, [plan.pregrasp] + plan.approach, ctx.should_cancel)

    ctx.phase(Phase.GRASPING)
    if cfg.use_gripper:
        ctx.robot.close_gripper(ctx.should_cancel)
        # TODO[lnfu] check the direction once the gripper range is known.
        if ctx.robot.gripper_position() <= cfg.closed_threshold:
            raise GraspFailed("gripper closed completely: nothing grasped")
    else:
        ctx.log.warn("use_gripper is false: not closing the gripper")

    ctx.phase(Phase.LIFTING)
    ctx.robot.move_arm(plan.approach[-1], [plan.lift], ctx.should_cancel)
    return center


def _solvers(cfg, urdf):
    if urdf is None:
        raise GraspFailed("no /robot_description received yet")
    try:
        return make_solvers(
            urdf,
            cfg.base_dof_list,
            free_yaw=cfg.free_yaw,
            grasp_yaw=cfg.grasp_wrist["wrist_yaw_joint"],
            max_aim_yaw=np.deg2rad(cfg.max_aim_yaw_deg),
            max_base_translation=cfg.max_base_translation,
            max_base_rotation=np.deg2rad(cfg.max_base_rotation_deg),
            ee_frame=cfg.ee_frame,
            base_frame=cfg.base_frame,
        )
    except ValueError as e:
        raise GraspFailed(f"cannot build IK model: {e}") from e


def _detect(goal, cfg, ctx, rotation):
    """The center of goal.prompt in the base frame (the arm must be still)."""
    seen = ctx.locator.locate(goal.prompt, ctx.should_cancel)
    width = object_width(seen.points, rotation)
    ctx.log.info(f"{seen.label}: {len(seen.points)} points, width {width * 100:.1f} cm")
    if width > cfg.gripper_max_opening:
        raise GraspFailed(
            f"object is {width * 100:.1f} cm wide, wider than the gripper opening"
        )
    return center_and_width(seen.points)[0]


def object_width(points: np.ndarray, rotation: np.ndarray) -> float:
    """Extent of the object points (base frame) along the gripper's closing axis, the
    y axis of grasp_center_link (`rotation`).

    TODO[lnfu] confirm the closing axis on the robot (finger joints rotate about z,
    offset along y).
    """
    return center_and_width(points @ rotation, width_axis=1)[1]
