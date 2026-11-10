from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
import pinocchio as pin

from ...errors import GraspFailed
from ...geometry.points import center_and_width, transform_points
from ..primitives import approach_with_base
from .plan import fixed_grasp_rotation, make_solvers, plan_grasp


class Phase(IntEnum):
    # Values match the feedback phases in Grasp.action.
    MOVING_TO_START = 0
    DETECTING = 1
    APPROACHING = 2
    GRASPING = 3
    LIFTING = 4


@dataclass(frozen=True)
class Observation:
    label: str
    score: float
    points: np.ndarray  # (N, 3), base frame
    truncated: bool  # the mask touches the image border

    def transform(self, pose: pin.SE3) -> "Observation":
        return replace(self, points=transform_points(self.points, pose))


@dataclass(frozen=True)
class GraspGoal:
    prompt: str = ""
    point: np.ndarray | None = None  # base frame, no camera


def run(goal: GraspGoal, cfg, shared, ctx) -> np.ndarray:
    arm_solver, base_solver = _solvers(cfg, ctx.frames, ctx.robot.urdf)
    rotation = fixed_grasp_rotation(arm_solver, cfg.grasp_wrist)

    ctx.phase(Phase.MOVING_TO_START)
    ctx.robot.move_arm(ctx.robot.arm_joints(), [cfg.start_pose], ctx.cancel)
    ctx.cancel.sleep(shared.settle_time)
    current = ctx.robot.arm_joints()

    ctx.phase(Phase.DETECTING)
    center = goal.point if goal.point is not None else _detect(goal, cfg, ctx, rotation)
    _show_target(ctx, center)

    def replan(current, center):
        plan = plan_grasp(
            arm_solver,
            current,
            center,
            rotation,
            pregrasp_distance=cfg.pregrasp_distance,
            n_waypoints=cfg.n_waypoints,
            lift_height=cfg.lift_height,
            accept_position_error=cfg.ik.position_error,
            accept_orientation_error=np.deg2rad(cfg.ik.orientation_error_deg),
            base_solver=base_solver,
            base_posture={
                "arm_joint": cfg.base_stage_arm,
                "wrist_yaw_joint": cfg.grasp_wrist["wrist_yaw_joint"],
            },
        )
        ctx.report.show(center, plan.poses["approach"] + [plan.poses["lift"]])
        ctx.log.info(f"plan ok: base {plan.base or 'locked'}, center {center.round(3)}")
        return plan

    def relocate(plan, center):
        ctx.phase(Phase.DETECTING)
        if goal.point is None:
            center = _detect(goal, cfg, ctx, rotation)
        else:  # no camera: use the planned base motion
            moved = base_solver.base_motion({**plan.pregrasp, **plan.base})
            center = moved.actInv(pin.SE3(np.eye(3), center)).translation
        _show_target(ctx, center)
        ctx.phase(Phase.APPROACHING)
        return center

    plan = replan(current, center)
    ctx.phase(Phase.APPROACHING)
    plan, current, center = approach_with_base(
        ctx,
        plan,
        replan,
        relocate,
        current,
        center,
        max_moves=cfg.max_base_moves,
        gains=cfg.base_control,
        settle_time=shared.settle_time,
    )

    ctx.robot.open_gripper(ctx.cancel)
    ctx.robot.move_arm(current, [plan.pregrasp] + plan.approach, ctx.cancel)

    ctx.phase(Phase.GRASPING)
    if not ctx.robot.close_gripper(ctx.cancel):
        raise GraspFailed("gripper closed all the way: nothing grasped")

    ctx.phase(Phase.LIFTING)
    ctx.robot.move_arm(plan.approach[-1], [plan.lift], ctx.cancel)
    return center


def _solvers(cfg, frames, urdf):
    if urdf is None:
        raise GraspFailed("no /robot_description received yet")
    try:
        return make_solvers(
            urdf,
            cfg.base_dof_list,
            free_yaw=cfg.free_yaw,
            grasp_yaw=cfg.grasp_wrist["wrist_yaw_joint"],
            max_aim_yaw=np.deg2rad(cfg.max_aim_yaw_deg),
            max_base_translation=cfg.base_limits.translation,
            max_base_rotation=np.deg2rad(cfg.base_limits.rotation_deg),
            ee_frame=frames.ee,
            base_frame=frames.base,
        )
    except ValueError as e:
        raise GraspFailed(f"cannot build IK model: {e}") from e


def _detect(goal, cfg, ctx, rotation):
    seen = ctx.locate(goal.prompt, ctx.cancel)
    width = object_width(seen.points, rotation)
    ctx.log.info(f"{seen.label}: {len(seen.points)} points, width {width * 100:.1f} cm")
    if width > cfg.gripper_max_opening:
        raise GraspFailed(
            f"object is {width * 100:.1f} cm wide, wider than the gripper opening"
        )
    return center_and_width(seen.points)[0]


def _show_target(ctx, center):
    ctx.report.show(center)  # before planning: visible even if IK fails
    ctx.log.info(f"target in the base frame: {center.round(3)}")


def object_width(points: np.ndarray, rotation: np.ndarray) -> float:
    # TODO[lnfu] confirm on the robot that the gripper closes along its y axis.
    return center_and_width(points @ rotation, width_axis=1)[1]
