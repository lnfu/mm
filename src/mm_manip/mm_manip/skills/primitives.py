import time
from dataclasses import dataclass
from typing import Any, Callable

from ..errors import Canceled


class OutOfReach(RuntimeError):
    pass


@dataclass
class Context:
    """What a skill runs against.

    robot: arm_joints(), move_arm(start, points, cancel), move_base(x, y, theta,
        cancel, gains), open_gripper(cancel), close_gripper(cancel) -> True if
        holding something, urdf.
    locator: locate(prompt, cancel) -> object with .label and .points (base frame).
    observer: phase(phase) plus whatever the skill reports (targets, plans, ...).
    log: info(msg), warn(msg).
    """

    robot: Any
    locator: Any
    observer: Any
    log: Any
    should_cancel: Callable[[], bool]

    def phase(self, phase):
        """Enter `phase`; the place where a canceled goal stops."""
        if self.should_cancel():
            raise Canceled()
        self.observer.phase(phase)

    def sleep(self, seconds):
        sleep(seconds, self.should_cancel)


def sleep(seconds, should_cancel):
    """time.sleep that raises Canceled as soon as should_cancel() is true."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if should_cancel():
            raise Canceled()
        time.sleep(0.02)


def needs_base(plan) -> bool:
    return any(abs(v) > 1e-3 for v in plan.base.values())


def approach_with_base(
    ctx, plan, replan, relocate, current, center, *, max_moves, gains, settle_time
):
    """Move the base until the arm alone reaches the target; look again after each move.

    plan: the first plan; its `.base` is the base motion it needs ({} or ~0 when
        the arm alone reaches).
    replan(current, center) -> plan, the same kind as `plan`.
    relocate(plan, center) -> center seen from the moved base: detect it again, or
        move it by the planned motion when there is no camera.

    The base never lands exactly where planned (wheel slip) and a first look near
    the image edge can be far off, so after a move the target may still be out of
    reach: the base may then move again, up to max_moves times in all.
    Returns (plan, arm joints, center) once no base motion is left.
    Raises OutOfReach after max_moves moves.
    """
    moves = 0
    while needs_base(plan):
        if moves == max_moves:
            raise OutOfReach(f"still out of arm reach after {moves} base moves")
        ctx.robot.move_base(
            plan.base.get("base_x", 0.0),
            plan.base.get("base_y", 0.0),
            plan.base.get("base_theta", 0.0),
            ctx.should_cancel,
            gains,
        )
        moves += 1
        # TODO[lnfu]
        ctx.sleep(settle_time)
        current = ctx.robot.arm_joints()
        center = relocate(plan, center)
        plan = replan(current, center)
    return plan, current, center
