from dataclasses import fields, replace

import numpy as np
import pytest

from mm_ik.solver import IKSolver
from mm_manip.cancel import CancelToken
from mm_manip.config import ArticulateConfig, ServerConfig
from mm_manip.errors import ArticulateFailed, OutOfReach
from mm_manip.skills.articulate import skill as articulate
from mm_manip.skills.articulate.door import (
    Door,
    DoorType,
    HandleType,
    door_from_points,
    hinge_axis,
    into_handle,
    opening_arc,
    with_handle_offset,
)
from mm_manip.skills.articulate.skill import ArticulateGoal, Phase
from mm_manip.skills.primitives import Context

stretch4_urdf = pytest.importorskip("stretch4_urdf")

# A door 0.8 m ahead, 0.4 m wide, facing the robot.
NORMAL = np.array([-1.0, 0.0, 0.0])
CORNERS = {
    "top_left": np.array([0.8, 0.2, 1.0]),
    "bottom_left": np.array([0.8, 0.2, 0.4]),
    "top_right": np.array([0.8, -0.2, 1.0]),
    "bottom_right": np.array([0.8, -0.2, 0.4]),
}


def door(door_type=DoorType.REVOLUTE_LEFT, handle=(0.8, -0.15, 0.7), **kw):
    values = dict(
        type=door_type,
        handle_type=HandleType.BAR_VERTICAL,
        handle=np.array(handle, float),
        corners=CORNERS,
        normal=NORMAL,
    )
    values.update(kw)
    return Door(**values)


def test_hinge_axis_is_on_the_hinge_side_at_handle_height():
    assert hinge_axis(door()) == pytest.approx([0.8, 0.2, 0.7])
    right = door(DoorType.REVOLUTE_RIGHT, handle=(0.8, 0.15, 0.7))
    assert hinge_axis(right) == pytest.approx([0.8, -0.2, 0.7])


def test_hinge_axis_from_a_slanted_edge_stays_vertical():
    corners = {**CORNERS, "top_left": np.array([0.82, 0.22, 1.0])}
    axis = hinge_axis(door(corners=corners))
    assert axis[2] == pytest.approx(0.7)
    assert axis[:2] == pytest.approx([0.81, 0.21], abs=2e-3)


@pytest.mark.parametrize(
    "door_type, handle, end",
    [
        (DoorType.REVOLUTE_LEFT, (0.8, -0.15, 0.7), (0.45, 0.2)),
        (DoorType.REVOLUTE_RIGHT, (0.8, 0.15, 0.7), (0.45, -0.2)),
    ],
)
def test_arc_swings_the_handle_toward_the_robot(door_type, handle, end):
    arc = opening_arc(door(door_type, handle), np.pi / 2, 10)
    points = np.array([p.translation for p in arc.poses])
    assert points[0] == pytest.approx(handle)
    assert points[:, 2] == pytest.approx(np.full(10, 0.7))
    radii = np.linalg.norm(points[:, :2] - arc.axis[:2], axis=1)
    assert radii == pytest.approx(np.full(10, 0.35))
    assert points[1, 0] < points[0, 0]
    assert points[-1, :2] == pytest.approx(end)
    assert arc.angles[-1] == pytest.approx(np.pi / 2)


def test_gripper_points_into_the_door_and_turns_with_it():
    arc = opening_arc(door(), np.pi / 2, 3)
    assert arc.poses[0].rotation[:, 0] == pytest.approx([1, 0, 0])
    assert arc.poses[0].rotation[:, 1] == pytest.approx([0, 1, 0])
    assert arc.sign == -1
    assert arc.poses[-1].rotation[:, 0] == pytest.approx([0, -1, 0])


def test_horizontal_bar_rolls_the_gripper():
    arc = opening_arc(
        door(), np.pi / 2, 3, roll=articulate.HANDLE_ROLL[HandleType.BAR_HORIZONTAL]
    )
    assert abs(arc.poses[0].rotation[:, 1] @ [0, 0, 1]) == pytest.approx(1)


def test_door_from_points():
    d = door_from_points([0.8, -0.15, 0.7], [0.8, 0.2, 0.3])
    assert d.type == DoorType.REVOLUTE_LEFT
    assert d.normal == pytest.approx(NORMAL)
    assert hinge_axis(d) == pytest.approx([0.8, 0.2, 0.7])
    assert door_from_points([0.8, 0.15, 0.7], [0.8, -0.2, 0.3]).type == (
        DoorType.REVOLUTE_RIGHT
    )


def test_handle_offset_moves_the_handle_out_of_the_door():
    assert with_handle_offset(door(), 0.03).handle == pytest.approx([0.77, -0.15, 0.7])


def test_into_handle_moves_the_front_surface_into_the_door():
    moved, protrusion, depth = into_handle(door(handle=(0.77, -0.15, 0.7)), 0.015)
    assert protrusion == pytest.approx(0.03)
    assert depth == pytest.approx(0.015)
    assert moved.handle == pytest.approx([0.785, -0.15, 0.7])


def test_into_handle_stops_short_of_the_door():
    moved, _, depth = into_handle(door(handle=(0.79, -0.15, 0.7)), 0.015)
    assert depth == pytest.approx(0.005)
    assert moved.handle == pytest.approx([0.795, -0.15, 0.7])


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
def handle(urdf):
    return IKSolver(urdf).fk({**START, "arm_joint": 0.35}).translation.copy()


class FakeRobot:
    def __init__(self, urdf):
        self.urdf = urdf
        self.joints = dict(START)
        self.paths = []
        self.holding = True
        self.gripper = []

    def arm_joints(self):
        return dict(self.joints)

    def follow_path(self, points, cancel, gains=None):
        self.paths.append(points)
        self.joints.update({k: points[-1][k] for k in START})

    def open_gripper(self, cancel):
        self.gripper.append(("open", len(self.paths)))

    def close_gripper(self, cancel):
        self.gripper.append(("close", len(self.paths)))
        return self.holding


class FakeLocator:
    def __init__(self, found):
        self.found = found

    def locate(self, prompt, cancel):
        return self.found


class FakeObserver:
    def __init__(self):
        self.phases, self.doors, self.arcs, self.ghosts = [], [], [], []

    def phase(self, phase):
        self.phases.append(phase)

    def door(self, door):
        self.doors.append(door)

    def arc(self, arc):
        self.arcs.append(arc)

    def ghost(self, waypoints):
        self.ghosts.append(waypoints)


class FakeLog:
    def info(self, message):
        pass

    def warn(self, message):
        pass


def make_case(urdf, found=None, **cfg):
    robot = FakeRobot(urdf)
    observer = FakeObserver()
    ctx = Context(robot, FakeLocator(found), observer, FakeLog(), CancelToken())
    defaults = dict(dry_run=False, start_pose=START, settle_time=0.0)
    values = {**defaults, **cfg}
    shared_names = {f.name for f in fields(ServerConfig)}
    shared = {k: v for k, v in values.items() if k in shared_names}
    own = {k: v for k, v in values.items() if k not in shared_names}
    return replace(ArticulateConfig(), **own), replace(ServerConfig(), **shared), ctx


def point_goal(handle, hinge_dy):
    return ArticulateGoal(handle=handle, hinge=handle + [0.0, hinge_dy, 0.0])


@pytest.mark.parametrize("hinge_dy", [0.30, -0.30])
def test_opens_a_door_given_by_points(urdf, handle, hinge_dy):
    cfg, shared, ctx = make_case(urdf)
    outcome = articulate.run(point_goal(handle, hinge_dy), cfg, shared, ctx)
    assert ctx.observer.phases == list(Phase)
    start, grasp, arc = ctx.robot.paths
    assert len(start) == 1 and len(grasp) == 1
    assert "base_x" not in grasp[0]
    assert {"base_x", "base_y", "base_theta"} <= set(arc[-1])
    assert 1 <= len(arc) <= cfg.n_waypoints - 1
    assert 0 < outcome.opened <= outcome.asked == pytest.approx(np.pi / 2)


def test_far_handle_moves_the_base_with_the_arm(urdf, handle):
    cfg, shared, ctx = make_case(urdf)
    far = handle + [0.3, 0.0, 0.0]
    articulate.run(point_goal(far, -0.30), cfg, shared, ctx)
    _, (grasp,), arc = ctx.robot.paths
    assert grasp["base_x"] > 0.1
    assert grasp["arm_joint"] == pytest.approx(cfg.approach_arm, abs=0.1)
    assert len(arc) >= 1


def test_dry_run_only_plans(urdf, handle):
    cfg, shared, ctx = make_case(urdf, dry_run=True)
    articulate.run(point_goal(handle, 0.30), cfg, shared, ctx)
    assert ctx.robot.paths == []
    assert len(ctx.observer.ghosts) == 1


def test_gripper_opens_before_reaching_and_closes_on_the_handle(urdf, handle):
    cfg, shared, ctx = make_case(urdf)
    articulate.run(point_goal(handle, 0.30), cfg, shared, ctx)
    assert ctx.robot.gripper == [("open", 1), ("close", 2)]


def test_missed_handle_fails_before_pulling(urdf, handle):
    cfg, shared, ctx = make_case(urdf)
    ctx.robot.holding = False
    with pytest.raises(ArticulateFailed, match="missed the handle"):
        articulate.run(point_goal(handle, 0.30), cfg, shared, ctx)
    assert len(ctx.robot.paths) == 2


@pytest.mark.parametrize(
    "found",
    [
        door(DoorType.PRISMATIC),
        door(DoorType.UNKNOWN),
        door(handle_type=HandleType.RECESSED),
        door(handle_type=HandleType.UNKNOWN),
    ],
)
def test_unsupported_doors_are_not_implemented(urdf, found):
    cfg, shared, ctx = make_case(urdf, found=found)
    with pytest.raises(NotImplementedError):
        articulate.run(ArticulateGoal(prompt="door"), cfg, shared, ctx)
    (seen,) = ctx.observer.doors
    assert (seen.type, seen.handle_type) == (found.type, found.handle_type)
    assert len(ctx.robot.paths) == 1


def test_handle_out_of_reach(urdf):
    cfg, shared, ctx = make_case(urdf)
    far = np.array([2.0, 0.0, 0.7])
    with pytest.raises(OutOfReach):
        articulate.run(point_goal(far, 0.3), cfg, shared, ctx)
