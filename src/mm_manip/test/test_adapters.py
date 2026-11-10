import numpy as np
import pytest

from mm_manip.cancel import CancelToken
from mm_manip.errors import MotionError
from mm_manip.skills.adapters import (
    DryRunLocate,
    DryRunRobot,
    WithoutGripper,
    base_motion,
)


class Robot:
    urdf = "<robot/>"

    def __init__(self, joints=None):
        self.joints = joints

    def arm_joints(self, max_age=None):
        if self.joints is None:
            raise MotionError("no joint states")
        return dict(self.joints)


class Report:
    def __init__(self):
        self.ghosts = []

    def ghost(self, waypoints):
        self.ghosts.append(waypoints)


class Seen:
    def __init__(self, point):
        self.point = np.asarray(point, float)

    def transform(self, pose):
        return Seen(pose.act(self.point))


ARM = {
    "lift_joint": 0.5,
    "arm_joint": 0.1,
    "wrist_yaw_joint": 0.0,
    "wrist_pitch_joint": 0.0,
    "wrist_roll_joint": 0.0,
}


def test_base_motion_translates_then_turns():
    pose = base_motion(0.2, 0.1, np.pi / 2)
    assert pose.translation == pytest.approx([0.2, 0.1, 0.0])
    assert pose.rotation @ [1, 0, 0] == pytest.approx([0, 1, 0])


def test_dry_run_robot_keeps_the_simulated_arm_and_base():
    robot = DryRunRobot(Robot(ARM), Report(), CancelToken())
    target = {**ARM, "arm_joint": 0.3}
    robot.move_arm(ARM, [target], CancelToken())
    assert robot.arm_joints() == target
    robot.follow_path([{**target, "base_x": 0.5}], CancelToken())
    robot.move_base(0.0, 0.0, np.pi / 2, CancelToken())
    assert robot.base.translation == pytest.approx([0.5, 0.0, 0.0])
    assert robot.base.rotation @ [1, 0, 0] == pytest.approx([0, 1, 0])
    assert robot.close_gripper(CancelToken()) and robot.urdf == "<robot/>"


def test_dry_run_robot_without_joint_states_needs_a_move_first():
    robot = DryRunRobot(Robot(None), Report(), CancelToken())
    with pytest.raises(MotionError):
        robot.arm_joints()
    robot.move_arm({}, [ARM], CancelToken())
    assert robot.arm_joints() == ARM


def test_dry_run_locate_looks_once_and_follows_the_simulated_base():
    looks = []

    def locate(prompt, cancel):
        looks.append(prompt)
        return Seen([1.0, 0.0, 0.5])

    robot = DryRunRobot(Robot(ARM), Report(), CancelToken())
    found = DryRunLocate(locate, robot)
    assert found("can", CancelToken()).point == pytest.approx([1.0, 0.0, 0.5])
    robot.move_base(0.4, 0.1, 0.0, CancelToken())
    assert found("can", CancelToken()).point == pytest.approx([0.6, -0.1, 0.5])
    assert looks == ["can"]


def test_without_gripper_passes_everything_else_through():
    class Log:
        def __init__(self):
            self.warnings = []

        def warn(self, message):
            self.warnings.append(message)

    log = Log()
    robot = WithoutGripper(Robot(ARM), log)
    assert robot.close_gripper(CancelToken())
    robot.open_gripper(CancelToken())
    assert robot.arm_joints() == ARM and robot.urdf == "<robot/>"
    assert len(log.warnings) == 2
