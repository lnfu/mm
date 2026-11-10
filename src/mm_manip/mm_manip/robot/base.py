import time

from ..cancel import CancelToken
from ..errors import MotionError
from ..motion.base_control import BaseGains, rotate_command, translate_command
from .driver import Driver
from .state import RobotState


class BaseMover:
    """Base moves on /cmd_vel, closed loop on /wheel_odom."""

    def __init__(self, driver: Driver, state: RobotState, rate: float = 20.0):
        self._driver = driver
        self._state = state
        self._period = 1.0 / rate

    def move(self, x: float, y: float, theta: float, cancel: CancelToken, gains=None):
        """Translate by (x, y), then turn by theta: the order of the IK base joints."""
        gains = gains or BaseGains()
        start = self._state.odom_pose(max_age=0.5)
        distance = (x**2 + y**2) ** 0.5
        timeout = 5.0 + 2.0 * (
            distance / gains.max_linear + abs(theta) / gains.max_angular
        )
        deadline = time.monotonic() + timeout
        with self._driver.streaming(arm=False, base=True):
            for command in (
                lambda p: translate_command(start, p, x, y, gains),
                lambda p: rotate_command(start, p, theta, gains),
            ):
                while True:
                    cancel.check()
                    if time.monotonic() > deadline:
                        raise MotionError(
                            f"base motion timed out after {timeout:.0f} s"
                        )
                    vx, vy, w, done = command(self._state.odom_pose(max_age=0.5))
                    if done:
                        break
                    self._driver.twist(vx, vy, w)
                    time.sleep(self._period)
                self._driver.twist(0.0, 0.0, 0.0)
