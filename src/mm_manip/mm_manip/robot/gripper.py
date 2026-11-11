import time

from control_msgs.action import FollowJointTrajectory

from ..cancel import POLL, CancelToken
from ..errors import Canceled, MotionError
from ..motion.gripper import GripperSettings, StallDetector
from .driver import Driver
from .state import RobotState


class Gripper:
    # stretch_driver ends a gripper goal only within 0.5 of the target. At a
    # mechanical stop or on an object it never gets there, so wait until the
    # fingers stop.

    def __init__(self, node, driver: Driver, state: RobotState, mode: str):
        self._node = node
        self._driver = driver
        self._state = state
        self._mode = mode
        self.settings = GripperSettings()

    def open(self, cancel: CancelToken):
        self._run(self.settings.open, cancel)

    def close(self, cancel: CancelToken) -> bool:
        """True if it holds something. The command stays on to keep the grip."""
        g = self.settings
        self._run(g.closed, cancel)
        finger = self._state.joint(g.state_joint)
        holding = abs(finger - g.closed_finger) > g.holding_margin
        self._node.get_logger().info(
            f"gripper closed at finger {finger:.3f} (empty {g.closed_finger:.3f}): "
            f"{'holding something' if holding else 'nothing in it'}"
        )
        return holding

    def _run(self, target: float, cancel: CancelToken) -> bool:
        """True at the target, False when the fingers stopped before it."""
        g = self.settings
        self._driver.ensure_mode(self._mode, cancel)
        goal = self._driver.send_point({g.joint: target}, cancel, what="gripper goal")
        result = goal.get_result_async()
        start = time.monotonic()
        stall = StallDetector(g, self._state.joint(g.state_joint), start)
        while True:
            if cancel.canceled:
                goal.cancel_goal_async()
                raise Canceled()
            if result.done():
                outcome = result.result().result
                if outcome.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
                    return True
                raise MotionError(
                    f"gripper to {target} failed ({outcome.error_code}): "
                    f"{outcome.error_string}"
                )
            now = time.monotonic()
            if stall.stopped(self._state.joint(g.state_joint), now):
                return False
            if now - start > g.timeout:
                raise MotionError(f"gripper did not reach {target} in {g.timeout} s")
            time.sleep(POLL)
