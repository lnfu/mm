from dataclasses import dataclass, field

import numpy as np

from mm_ik.solver import IKSolver

from ...errors import ArticulateFailed, OutOfReach
from ...motion.planning import PlanningError, arm_only, reach, solve_waypoints
from ...motion.velocity import BASE_JOINTS, BASE_THETA, BASE_X, BASE_Y
from .door import Arc

# Posture costs. The arm joints have 1e-3.
BASE_TRANSLATION_COST = 0.1


FREE_TURN_COST = 1e-3


PRESCRIBED_TURN_COST = 1.0


APPROACH_ARM_POSTURE_COST = 1e-2


APPROACH_TRANSLATION_COST = 1e-4


@dataclass(frozen=True)
class ArcPlan:
    grasp: dict  # arm joints and the base motion to the grasp
    path: list  # base joints are relative to the base at the grasp
    opened: float  # rad
    strategy: str
    base_travel: float = field(default=0.0)


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
        [{**arm_only(p), **{j: p[j] for j in BASE_JOINTS}} for p in path],
        float(angles[n - 1]),
        name,
        travel,
    )
