from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from mm_ik.solver import IKSolver
from mm_manip.cancel import CancelToken
from mm_manip.config import GoalConfig
from mm_manip.errors import Canceled, GraspFailed, OutOfReach
from mm_manip.skills.base import Context
from mm_manip.skills.grasp import skill as grasp
from mm_manip.skills.grasp.config import GraspConfig
from mm_manip.skills.grasp.skill import GraspGoal, Phase, object_width

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
    return IKSolver(urdf).fk({**START, "arm_joint": 0.35}).translation.copy()


class FakeRobot:
    def __init__(self, urdf):
        self.urdf = urdf
        self.joints = dict(START)
        self.base = np.zeros(3)
        self.arm_moves = []
        self.base_moves = []
        self.holding = True
        self.gripper = []

    def open_gripper(self, cancel):
        self.gripper.append("open")

    def close_gripper(self, cancel):
        self.gripper.append("close")
        return self.holding

    def arm_joints(self):
        return dict(self.joints)

    def move_arm(self, start, points, cancel):
        self.arm_moves.append(points)
        self.joints.update(points[-1])

    def move_base(self, x, y, theta, cancel, gains):
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
        self.errors = list(errors)
        self.looks = 0

    def locate(self, prompt, cancel):
        center = self.robot.to_base(self.world)
        if self.looks < len(self.errors):
            center = center + self.errors[self.looks]
        self.looks += 1
        offsets = np.array(
            [[0, dy, dz] for dy in (-0.03, 0, 0.03) for dz in (-0.05, 0, 0.05)]
        )
        return SimpleNamespace(label=prompt, points=center + offsets)


class FakeReport:
    def __init__(self):
        self.phases, self.shown, self.ghosts, self.kept = [], [], [], {}

    def phase(self, phase):
        self.phases.append(phase)

    def show(self, center, poses=()):
        self.shown.append((center, list(poses)))

    def ghost(self, waypoints):
        self.ghosts.append(waypoints)

    def keep(self, **values):
        self.kept.update(values)


class FakeLog:
    def info(self, message):
        pass

    def warn(self, message):
        pass


def looks(ctx) -> int:
    return ctx.locate.__self__.looks  # ctx.locate is FakeLocator.locate


def make_case(urdf, world, errors=(), cancel_after=None, **cfg):
    robot = FakeRobot(urdf)
    locator = FakeLocator(robot, world, errors)
    report = FakeReport()

    cancel = CancelToken(
        lambda: cancel_after is not None and len(report.phases) >= cancel_after
    )
    ctx = Context(robot, locator.locate, report, FakeLog(), cancel)
    defaults = dict(dry_run=False, start_pose=START, base_dofs="x,y", settle_time=0.0)
    values = {**defaults, **cfg}
    shared_names = {f.name for f in fields(GoalConfig)}
    shared = {k: v for k, v in values.items() if k in shared_names}
    own = {k: v for k, v in values.items() if k not in shared_names}
    return replace(GraspConfig(), **own), replace(GoalConfig(), **shared), ctx


def test_reachable_object_needs_no_base(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable)
    center = grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert ctx.robot.base_moves == []
    assert looks(ctx) == 1
    assert len(ctx.robot.arm_moves) == 3
    assert np.allclose(center, reachable)
    assert ctx.report.phases == [
        Phase.MOVING_TO_START,
        Phase.DETECTING,
        Phase.APPROACHING,
        Phase.GRASPING,
        Phase.LIFTING,
    ]


def test_far_object_moves_the_base_once_and_looks_again(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable + [0.0, -0.30, 0.0])
    grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert len(ctx.robot.base_moves) == 1
    assert looks(ctx) == 2


def test_wrong_first_look_gets_a_second_base_move(urdf, reachable):
    # Real case 2026-11-02, see NOTE.md.
    cfg, shared, ctx = make_case(
        urdf, reachable + [0.0, -0.30, 0.0], errors=[[0, 0.15, 0.07]]
    )
    grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert len(ctx.robot.base_moves) == 2
    assert looks(ctx) == 3
    end = ctx.robot.to_base(reachable + [0.0, -0.30, 0.0])
    assert np.allclose(ctx.report.shown[-1][0], end)


def test_out_of_reach_after_max_base_moves(urdf, reachable):
    cfg, shared, ctx = make_case(
        urdf,
        reachable + [0.0, -0.30, 0.0],
        errors=[[0, 0.15, 0.07]],
        max_base_moves=1,
    )
    with pytest.raises(OutOfReach):
        grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert len(ctx.robot.base_moves) == 1


def test_point_goal_uses_no_camera(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable)
    far = reachable + [0.0, -0.30, 0.0]
    center = grasp.run(GraspGoal(point=far), cfg, shared, ctx)
    assert looks(ctx) == 0
    assert len(ctx.robot.base_moves) == 1
    assert np.allclose(center, ctx.robot.to_base(far), atol=1e-3)


def test_dry_run_moves_nothing(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable + [0.0, -0.30, 0.0], dry_run=True)
    grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert ctx.robot.arm_moves == [] and ctx.robot.base_moves == []
    assert len(ctx.report.ghosts) == 1


def test_cancel_stops_at_the_next_phase(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable, cancel_after=1)
    with pytest.raises(Canceled):
        grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert ctx.report.phases == [Phase.MOVING_TO_START]
    assert looks(ctx) == 0


def test_too_wide_object_is_rejected(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable, gripper_max_opening=0.05)
    with pytest.raises(GraspFailed, match="wider than the gripper"):
        grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert len(ctx.robot.arm_moves) == 1


def test_object_width_along_closing_axis():
    R = np.eye(3)
    pts = np.column_stack(
        [np.linspace(0, 0.3, 200), np.linspace(-0.03, 0.03, 200), np.zeros(200)]
    )
    assert object_width(pts, R) == pytest.approx(0.06, abs=5e-3)


def test_gripper_opens_before_the_approach_and_closes_before_the_lift(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable)
    grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert ctx.robot.gripper == ["open", "close"]


def test_empty_gripper_fails_before_lifting(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable)
    ctx.robot.holding = False
    with pytest.raises(GraspFailed, match="nothing grasped"):
        grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert ctx.report.phases[-1] == Phase.GRASPING
    assert len(ctx.robot.arm_moves) == 2


def test_without_gripper_nothing_is_commanded(urdf, reachable):
    cfg, shared, ctx = make_case(urdf, reachable, use_gripper=False)
    grasp.run(GraspGoal(prompt="can"), cfg, shared, ctx)
    assert ctx.robot.gripper == []
