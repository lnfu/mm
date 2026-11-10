import math
import time

from mm_ik.joints import BASE_JOINTS, BASE_THETA, BASE_X, BASE_Y

from ..cancel import CancelToken
from ..errors import MotionError
from ..motion.base_control import displacement
from ..motion.velocity import PathTracker, TrackingGains
from .driver import Driver
from .state import RobotState

# /joint_vel: lift and arm get a velocity (m/s). The wrist joints get a
# distance (rad), because the driver does move_by(value * duration).
PRISMATIC = ("lift_joint", "arm_joint")


class PathFollower:
    """Velocity paths through the waypoints, without a stop at each one."""

    def __init__(
        self, driver: Driver, state: RobotState, rate: float, settle_timeout: float
    ):
        self._driver = driver
        self._state = state
        self._period = 1.0 / rate
        self._settle_timeout = settle_timeout

    def follow(self, points: list, cancel: CancelToken, gains: TrackingGains = None):
        """Base joints in `points`: travel from the start, in the start base frame."""
        if not points:
            raise ValueError("no waypoints")
        self._driver.ensure_mode("velocity", cancel)
        with_base = any(j in points[-1] for j in BASE_JOINTS)
        if with_base:
            points = [{**dict.fromkeys(BASE_JOINTS, 0.0), **p} for p in points]
        base_start = self._state.odom_pose(max_age=0.5) if with_base else None

        def measured():
            joints = self._state.arm_joints(max_age=0.5)
            if with_base:
                pose = self._state.odom_pose(max_age=0.5)
                joints[BASE_X], joints[BASE_Y], joints[BASE_THETA] = displacement(
                    base_start, pose
                )
            return joints

        tracker = PathTracker(measured(), points, gains or TrackingGains())
        arm = [j for j in tracker.joints if j not in BASE_JOINTS]
        deadline = tracker.duration + self._settle_timeout
        t0 = time.monotonic()
        with self._driver.streaming(arm=True, base=with_base):
            while True:
                cancel.check()
                t = time.monotonic() - t0
                current = measured()
                velocity, error, done = tracker.command(t, current)
                if done:
                    return
                if t > deadline:
                    far = {j: round(e, 4) for j, e in error.items() if abs(e) > 1e-3}
                    raise MotionError(f"path did not arrive, error {far}")
                ahead, _ = tracker.desired(t + self._period)
                self._driver.joint_velocities(
                    {
                        j: velocity[j] if j in PRISMATIC else ahead[j] - current[j]
                        for j in arm
                    }
                )
                if with_base:
                    # The path is in the start base frame, /cmd_vel in the current one.
                    turn = current[BASE_THETA]
                    vx, vy = velocity[BASE_X], velocity[BASE_Y]
                    self._driver.twist(
                        math.cos(turn) * vx + math.sin(turn) * vy,
                        -math.sin(turn) * vx + math.cos(turn) * vy,
                        velocity[BASE_THETA],
                    )
                time.sleep(self._period)
