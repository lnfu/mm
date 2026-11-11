from ..cancel import CancelToken
from ..motion.trajectory import MIN_JOINT_CHANGE, moving_joints, split_long_moves
from .driver import Driver


class FjtArm:
    """Arm moves point by point through FollowJointTrajectory."""

    def __init__(
        self, node, driver: Driver, mode: str, max_step: dict, no_split_timeout: float
    ):
        self._node = node
        self._driver = driver
        self._mode = mode
        self._max_step = max_step
        self._no_split_timeout = no_split_timeout
        self._split = None  # last answer, to log only changes

    def move(self, start: dict, points: list, cancel: CancelToken):
        self._driver.ensure_mode(self._mode, cancel)
        if self._needs_split(cancel):
            points = split_long_moves(start, points, self._max_step)
        # stretch_driver stops at each point and ignores time_from_start.
        # A lift or arm target at the current position never completes.
        # Thus, send only the joints that move.
        for point in moving_joints(start, points, MIN_JOINT_CHANGE):
            positions = {self._driver.driver_name(j): v for j, v in point.items()}
            goal = self._driver.send_point(positions, cancel)
            self._driver.wait_point(goal, cancel)

    def _needs_split(self, cancel: CancelToken) -> bool:
        # stretch_driver aborts a point after action_timeout (default 3 s).
        # Read it on each move, because it can change at run time.
        action_timeout = self._driver.action_timeout(cancel)
        split = action_timeout is None or action_timeout < self._no_split_timeout
        if split != self._split:
            self._split = split
            self._node.get_logger().info(
                f"stretch_driver action_timeout {action_timeout} s: "
                f"{'splitting long moves' if split else 'one point per move'}"
            )
        return split
