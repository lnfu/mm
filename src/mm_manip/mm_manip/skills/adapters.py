"""dry_run and use_gripper false as Robot and locate wrappers. See NOTE.md."""

import numpy as np
import pinocchio as pin

from mm_ik.joints import ARM_JOINT_NAMES, BASE_JOINTS, BASE_THETA, BASE_X, BASE_Y

from ..cancel import CancelToken
from ..errors import MotionError


def base_motion(x: float, y: float, theta: float) -> pin.SE3:
    """The base after (x, y, theta), in the base frame at the start."""
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pin.SE3(rotation, np.array([x, y, 0.0]))


class DryRunRobot:
    """Moves nothing. Keeps simulated joints and base pose, and shows each move
    as a ghost."""

    def __init__(self, robot, report, cancel: CancelToken):
        self._robot = robot
        self._report = report
        self._cancel = cancel
        self.base = pin.SE3.Identity()  # in the base frame at the goal start
        try:
            self._joints = robot.arm_joints()
        except MotionError:
            self._joints = None

    @property
    def urdf(self):
        return self._robot.urdf

    def arm_joints(self, max_age: float = None) -> dict:
        if self._joints is None:
            raise MotionError("dry_run: no arm joints yet")
        return dict(self._joints)

    def move_arm(self, start: dict, points: list, cancel: CancelToken):
        self._go([{j: p[j] for j in ARM_JOINT_NAMES} for p in points])

    def follow_path(self, points: list, cancel: CancelToken, gains=None):
        self._go([{j: p[j] for j in ARM_JOINT_NAMES} for p in points])
        end = points[-1]
        if any(j in end for j in BASE_JOINTS):
            self._move(
                end.get(BASE_X, 0.0), end.get(BASE_Y, 0.0), end.get(BASE_THETA, 0.0)
            )

    def move_base(self, x, y, theta, cancel: CancelToken, gains=None):
        self._move(x, y, theta)

    def open_gripper(self, cancel: CancelToken):
        pass

    def close_gripper(self, cancel: CancelToken) -> bool:
        return True

    def stop(self):
        pass

    def _go(self, waypoints: list):
        self._report.ghost(waypoints)
        self._joints = dict(waypoints[-1])

    def _move(self, x, y, theta):
        self.base = self.base * base_motion(x, y, theta)


class DryRunLocate:
    """Looks once; later looks give that result as seen from the simulated base."""

    def __init__(self, locate, robot: DryRunRobot):
        self._locate = locate
        self._robot = robot
        self._seen = None

    def __call__(self, prompt: str, cancel: CancelToken):
        if self._seen is None:
            self._seen = self._locate(prompt, cancel)
        return self._seen.transform(self._robot.base.inverse())


class WithoutGripper:
    """The robot with the gripper left alone: closing counts as holding."""

    def __init__(self, robot, log):
        self._robot = robot
        self._log = log

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def open_gripper(self, cancel: CancelToken):
        self._log.warn("use_gripper is false: not opening the gripper")

    def close_gripper(self, cancel: CancelToken) -> bool:
        self._log.warn("use_gripper is false: not closing the gripper")
        return True
