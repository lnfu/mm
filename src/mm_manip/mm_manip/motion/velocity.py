from dataclasses import dataclass, field
from itertools import pairwise

from mm_ik.joints import BASE_THETA, BASE_X, BASE_Y

# m/s or rad/s
DEFAULT_MAX_VELOCITY = {
    BASE_X: 0.05,
    BASE_Y: 0.05,
    BASE_THETA: 0.2,
    "lift_joint": 0.08,
    "arm_joint": 0.08,
    "wrist_yaw_joint": 0.6,
    "wrist_pitch_joint": 0.6,
    "wrist_roll_joint": 0.6,
}
# m or rad, at the last waypoint
DEFAULT_TOLERANCE = {
    BASE_X: 0.01,
    BASE_Y: 0.01,
    BASE_THETA: 0.02,
    "lift_joint": 0.003,
    "arm_joint": 0.003,
    "wrist_yaw_joint": 0.01,
    "wrist_pitch_joint": 0.01,
    "wrist_roll_joint": 0.01,
}


@dataclass(frozen=True)
class TrackingGains:
    max_velocity: dict = field(default_factory=lambda: dict(DEFAULT_MAX_VELOCITY))
    tolerance: dict = field(default_factory=lambda: dict(DEFAULT_TOLERANCE))
    gain: float = 2.0  # 1/s
    min_segment: float = 0.3  # s


class PathTracker:
    def __init__(self, start: dict, points: list, gains: TrackingGains):
        if not points:
            raise ValueError("no waypoints")
        self.joints = list(points[-1])
        self.gains = gains
        self._knots = [{j: start[j] for j in self.joints}] + [dict(p) for p in points]
        self._times = [0.0]
        for a, b in pairwise(self._knots):
            slowest = max(abs(b[j] - a[j]) / gains.max_velocity[j] for j in self.joints)
            self._times.append(self._times[-1] + max(slowest, gains.min_segment))

    @property
    def duration(self) -> float:
        return self._times[-1]

    def desired(self, t: float):
        """(position, velocity) at t s from the start."""
        if t >= self.duration:
            end = self._knots[-1]
            return dict(end), dict.fromkeys(self.joints, 0.0)
        t = max(t, 0.0)
        i = next(k for k in range(1, len(self._times)) if t < self._times[k])
        a, b = self._knots[i - 1], self._knots[i]
        span = self._times[i] - self._times[i - 1]
        s = (t - self._times[i - 1]) / span
        position = {j: a[j] + s * (b[j] - a[j]) for j in self.joints}
        velocity = {j: (b[j] - a[j]) / span for j in self.joints}
        return position, velocity

    def command(self, t: float, current: dict):
        """(velocity command, position error, done). Done: path ended, all in tolerance."""
        position, velocity = self.desired(t)
        error = {j: position[j] - current[j] for j in self.joints}
        done = t >= self.duration and all(
            abs(error[j]) <= self.gains.tolerance[j] for j in self.joints
        )
        command = {}
        for j in self.joints:
            limit = self.gains.max_velocity[j]
            v = velocity[j] + self.gains.gain * error[j]
            command[j] = min(max(v, -limit), limit)
        return command, error, done
