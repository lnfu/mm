"""The robot behind the skills' Robot protocol. See NOTE.md."""

from ..cancel import CancelToken
from ..motion.base_control import BaseGains, Pose2D
from ..motion.gripper import GripperSettings
from ..motion.trajectory import DEFAULT_MAX_STEP
from ..motion.velocity import TrackingGains
from .base import BaseMover
from .driver import Driver
from .fjt_arm import FjtArm
from .gripper import Gripper
from .path import PathFollower
from .state import RobotState


class Robot:
    def __init__(
        self,
        node,
        fjt_action: str,
        driver_joint_names: dict,
        max_step: dict = DEFAULT_MAX_STEP,
        no_split_action_timeout: float = 10.0,
        fjt_mode: str = "navigation",
        control_rate: float = 30.0,
        path_settle_timeout: float = 5.0,
        callback_group=None,
    ):
        self._state = RobotState(node, driver_joint_names, callback_group)
        self._driver = Driver(
            node, self._state, fjt_action, driver_joint_names, callback_group
        )
        self._arm = FjtArm(
            node, self._driver, fjt_mode, max_step, no_split_action_timeout
        )
        self._path = PathFollower(
            self._driver, self._state, control_rate, path_settle_timeout
        )
        self._base = BaseMover(self._driver, self._state)
        self._gripper = Gripper(node, self._driver, self._state, fjt_mode)

    @property
    def urdf(self) -> str | None:
        return self._state.urdf

    def arm_joints(self, max_age: float = None) -> dict:
        return self._state.arm_joints(max_age)

    def odom_pose(self, max_age: float) -> Pose2D:
        return self._state.odom_pose(max_age)

    def move_arm(self, start: dict, points: list, cancel: CancelToken):
        self._arm.move(start, points, cancel)

    def follow_path(
        self, points: list, cancel: CancelToken, gains: TrackingGains = None
    ):
        self._path.follow(points, cancel, gains)

    def move_base(self, x, y, theta, cancel: CancelToken, gains: BaseGains = None):
        self._base.move(x, y, theta, cancel, gains)

    def set_gripper(self, settings: GripperSettings):
        self._gripper.settings = settings

    def open_gripper(self, cancel: CancelToken):
        self._gripper.open(cancel)

    def close_gripper(self, cancel: CancelToken) -> bool:
        return self._gripper.close(cancel)

    def stop(self):
        self._driver.stop()
