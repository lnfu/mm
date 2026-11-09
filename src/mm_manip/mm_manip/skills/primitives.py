from dataclasses import dataclass
from typing import Any

from ..cancel import CancelToken
from ..errors import SkillError


class OutOfReach(SkillError):
    pass


@dataclass
class Context:
    robot: Any
    locator: Any
    observer: Any
    log: Any
    cancel: CancelToken

    def phase(self, phase):
        self.cancel.check()
        self.observer.phase(phase)


def needs_base(plan) -> bool:
    return any(abs(v) > 1e-3 for v in plan.base.values())


def approach_with_base(
    ctx, plan, replan, relocate, current, center, *, max_moves, gains, settle_time
):
    """Move the base until the arm alone can reach. Look again after each move."""
    moves = 0
    while needs_base(plan):
        if moves == max_moves:
            raise OutOfReach(f"still out of arm reach after {moves} base moves")
        ctx.robot.move_base(
            plan.base.get("base_x", 0.0),
            plan.base.get("base_y", 0.0),
            plan.base.get("base_theta", 0.0),
            ctx.cancel,
            gains,
        )
        moves += 1
        ctx.cancel.sleep(settle_time)
        current = ctx.robot.arm_joints()
        center = relocate(plan, center)
        plan = replan(current, center)
    return plan, current, center
