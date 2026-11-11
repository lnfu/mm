from dataclasses import dataclass


@dataclass(frozen=True)
class GripperSettings:
    joint: str = "gripper_joint"  # FollowJointTrajectory command
    state_joint: str = "gripper_finger_left_joint"  # /joint_states
    open: float = 80.0  # driver units, past the mechanical stop
    closed: float = -10.0  # driver units, past the mechanical stop
    closed_finger: float = 0.0  # rad, finger reading when closed on nothing
    holding_margin: float = 0.05  # rad
    still: float = 0.005  # rad
    stall_time: float = 0.3  # s
    timeout: float = 5.0  # s


class StallDetector:
    """The fingers stopped: still for stall_time, after they moved or after 1 s."""

    def __init__(self, settings: GripperSettings, position: float, now: float):
        self._settings = settings
        self._start = now
        self._last, self._still_since, self._moved = position, now, False

    def stopped(self, position: float, now: float) -> bool:
        if abs(position - self._last) > self._settings.still:
            self._last, self._still_since, self._moved = position, now, True
            return False
        still_long = now - self._still_since >= self._settings.stall_time
        return still_long and (self._moved or now - self._start > 1.0)
