from collections.abc import Callable
from enum import IntEnum
from typing import Any

from ..config import GoalConfig, StartupConfig


class SkillAction:
    """One skill on its ROS action, ~/<config.PREFIX>. The server makes one at start."""

    action: type
    config: type
    phases: type[IntEnum]

    def __init__(self, node, runtime, startup: StartupConfig):
        self.node = node
        self.runtime = runtime

    @property
    def name(self) -> str:
        return self.config.PREFIX

    def reject(self, goal) -> str | None:
        """Why the goal is not accepted, or None."""
        modes = (self.action.Goal.MODE_PROMPT, self.action.Goal.MODE_POINT)
        if goal.mode not in modes:
            return f"unknown mode {goal.mode}"
        if goal.mode == self.action.Goal.MODE_PROMPT and not goal.prompt.strip():
            return "MODE_PROMPT needs a prompt"
        return None

    def goal(self, request):
        raise NotImplementedError

    def locator(self, cfg) -> Callable[[str, Any], Any]:
        raise NotImplementedError

    def run(self, goal, cfg, shared: GoalConfig, ctx):
        raise NotImplementedError

    def message(self, outcome, shared: GoalConfig) -> str:
        raise NotImplementedError

    def fill_result(self, result, outcome, kept: dict) -> None:
        """outcome is None if the goal failed."""


def motion_message(shared: GoalConfig, done: str) -> str:
    if shared.dry_run:
        return "dry_run: plan ok"
    if shared.use_gripper:
        return done
    return "motion done (use_gripper is false, gripper not used)"
