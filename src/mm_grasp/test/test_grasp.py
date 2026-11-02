from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mm_grasp import grasp
from mm_grasp.config import GraspConfig
from mm_grasp.errors import Canceled
from mm_grasp.grasp import GraspGoal, Phase
from mm_grasp.primitives import Context, OutOfReach
from mm_ik.solver import IKSolver

stretch4_urdf = pytest.importorskip("stretch4_urdf")

START = {
    "lift_joint": 0.6,
    "arm_joint": 0.1,
    "wrist_yaw_joint": 0.0,
    "wrist_pitch_joint": 0.0,
    "wrist_roll_joint": 0.0,
}


@pytest.fixture(scope="module")
def urdf():
    return stretch4_urdf.get_urdf("SE4", "francis", "eoa_wrist_dw4_tool_sg4")


@pytest.fixture(scope="module")
def reachable(urdf):
    """An object center the arm alone reaches from START, in base_link."""
    return IKSolver(urdf).fk({**START, "arm_joint": 0.35}).translation.copy()


class FakeRobot:
    def __init__(self, urdf):
        self.urdf = urdf
        self.joints = dict(START)
        self.base = np.zeros(3)  # x, y, yaw in the world
        self.arm_moves = []
        self.base_moves = []

    def arm_joints(self):
        return dict(self.joints)

    def move_arm(self, start, points, should_cancel):
        self.arm_moves.append(points)
        self.joints.update(points[-1])

    def move_base(self, x, y, theta, should_cancel, gains):
        self.base_moves.append((x, y, theta))
        c, s = np.cos(self.base[2]), np.sin(self.base[2])
        self.base += [c * x - s * y, s * x + c * y, theta]

    def to_base(self, world):
        c, s = np.cos(self.base[2]), np.sin(self.base[2])
        d = world[:2] - self.base[:2]
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], world[2]])


class FakeLocator:
    def __init__(self, robot, world, errors=()):
        self.robot = robot
        self.world = world
        self.errors = list(errors)  # added to the center on the first looks
        self.looks = 0

    def locate(self, prompt, should_cancel):
        center = self.robot.to_base(self.world)
        if self.looks < len(self.errors):
            center = center + self.errors[self.looks]
        self.looks += 1
        offsets = np.array(
            [[0, dy, dz] for dy in (-0.03, 0, 0.03) for dz in (-0.05, 0, 0.05)]
        )
        return SimpleNamespace(label=prompt, points=center + offsets)


class FakeObserver:
    def __init__(self):
        self.phases, self.targets, self.plans, self.ghosts = [], [], [], []

    def phase(self, phase):
        self.phases.append(phase)

    def target(self, center):
        self.targets.append(center)

    def plan(self, center, plan):
        self.plans.append(plan)

    def ghost(self, waypoints):
        self.ghosts.append(waypoints)


class FakeLog:
    def info(self, message):
        pass

    def warn(self, message):
        pass


def make_case(urdf, world, errors=(), cancel_after=None, **cfg):
    robot = FakeRobot(urdf)
    locator = FakeLocator(robot, world, errors)
    observer = FakeObserver()

    def should_cancel():
        return cancel_after is not None and len(observer.phases) >= cancel_after

    ctx = Context(robot, locator, observer, FakeLog(), should_cancel)
    defaults = dict(dry_run=False, start_pose=START, base_dofs="x,y", settle_time=0.0)
    return replace(GraspConfig(), **{**defaults, **cfg}), ctx


def test_reachable_object_needs_no_base(urdf, reachable):
    cfg, ctx = make_case(urdf, reachable)
    center = grasp.run(GraspGoal(prompt="can"), cfg, ctx)
    assert ctx.robot.base_moves == []
    assert ctx.locator.looks == 1
    assert len(ctx.robot.arm_moves) == 3  # start pose, pre-grasp + approach, lift
    assert np.allclose(center, reachable)
    assert ctx.observer.phases == [
        Phase.MOVING_TO_START,
        Phase.DETECTING,
        Phase.APPROACHING,
        Phase.GRASPING,
        Phase.LIFTING,
    ]


def test_far_object_moves_the_base_once_and_looks_again(urdf, reachable):
    cfg, ctx = make_case(urdf, reachable + [0.0, -0.30, 0.0])
    grasp.run(GraspGoal(prompt="can"), cfg, ctx)
    assert len(ctx.robot.base_moves) == 1
    assert ctx.locator.looks == 2


def test_wrong_first_look_gets_a_second_base_move(urdf, reachable):
    """2026-11-02: a can far to the right, first seen 15 cm too far left (cut off at
    the image edge). The first base move leaves it out of arm reach; the base must
    move again instead of failing."""
    cfg, ctx = make_case(urdf, reachable + [0.0, -0.30, 0.0], errors=[[0, 0.15, 0.07]])
    grasp.run(GraspGoal(prompt="can"), cfg, ctx)
    assert len(ctx.robot.base_moves) == 2
    assert ctx.locator.looks == 3
    end = ctx.robot.to_base(reachable + [0.0, -0.30, 0.0])
    assert np.allclose(ctx.observer.targets[-1], end)


def test_out_of_reach_after_max_base_moves(urdf, reachable):
    cfg, ctx = make_case(
        urdf,
        reachable + [0.0, -0.30, 0.0],
        errors=[[0, 0.15, 0.07]],
        max_base_moves=1,
    )
    with pytest.raises(OutOfReach):
        grasp.run(GraspGoal(prompt="can"), cfg, ctx)
    assert len(ctx.robot.base_moves) == 1


def test_point_goal_uses_no_camera(urdf, reachable):
    cfg, ctx = make_case(urdf, reachable)
    far = reachable + [0.0, -0.30, 0.0]
    center = grasp.run(GraspGoal(point=far), cfg, ctx)
    assert ctx.locator.looks == 0
    assert len(ctx.robot.base_moves) == 1
    assert np.allclose(center, ctx.robot.to_base(far), atol=1e-3)


def test_dry_run_moves_nothing(urdf, reachable):
    cfg, ctx = make_case(urdf, reachable + [0.0, -0.30, 0.0], dry_run=True)
    grasp.run(GraspGoal(prompt="can"), cfg, ctx)
    assert ctx.robot.arm_moves == [] and ctx.robot.base_moves == []
    assert len(ctx.observer.ghosts) == 1


def test_cancel_stops_at_the_next_phase(urdf, reachable):
    cfg, ctx = make_case(urdf, reachable, cancel_after=1)
    with pytest.raises(Canceled):
        grasp.run(GraspGoal(prompt="can"), cfg, ctx)
    assert ctx.observer.phases == [Phase.MOVING_TO_START]
    assert ctx.locator.looks == 0


def test_too_wide_object_is_rejected(urdf, reachable):
    cfg, ctx = make_case(urdf, reachable, gripper_max_opening=0.05)
    with pytest.raises(grasp.GraspFailed, match="wider than the gripper"):
        grasp.run(GraspGoal(prompt="can"), cfg, ctx)
    assert len(ctx.robot.arm_moves) == 1  # only the start pose
