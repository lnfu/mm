from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..cancel import CancelToken


class Robot(Protocol):
    urdf: str | None

    def arm_joints(self, max_age: float = None) -> dict: ...

    def move_arm(self, start: dict, points: list, cancel: CancelToken) -> None: ...

    def move_base(self, x: float, y: float, theta: float, cancel, gains=None): ...

    def follow_path(self, points: list, cancel: CancelToken, gains=None) -> None: ...

    def open_gripper(self, cancel: CancelToken) -> None: ...

    def close_gripper(self, cancel: CancelToken) -> bool: ...


class Report(Protocol):
    def phase(self, phase) -> None: ...

    def show(self, center, poses=()) -> None: ...

    def ghost(self, waypoints: list) -> None: ...

    def keep(self, **values) -> None:
        """Values for the action result, also when the goal fails later."""


class Log(Protocol):
    def info(self, message: str) -> None: ...

    def warn(self, message: str) -> None: ...


@dataclass
class Context:
    robot: Robot
    locate: Callable[[str, CancelToken], Any]
    report: Report
    log: Log
    cancel: CancelToken

    def phase(self, phase):
        self.cancel.check()
        self.report.phase(phase)
