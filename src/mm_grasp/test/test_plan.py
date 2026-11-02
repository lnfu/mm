import numpy as np
import pytest

from mm_grasp.plan import (
    FREE_YAW_ORIENTATION_COST,
    PlanningError,
    fixed_grasp_rotation,
    grasp_poses,
    make_solvers,
    moving_joints,
    object_width,
    plan_grasp,
    reach,
    split_long_moves,
)


def test_straight_line_approach():
    yaw = np.deg2rad(30)
    R = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    center = np.array([0.6, 0.1, 0.8])
    poses = grasp_poses(
        center, R, pregrasp_distance=0.1, n_waypoints=5, lift_height=0.1
    )

    pre = poses["pregrasp"].translation
    assert np.linalg.norm(center - pre) == pytest.approx(0.1)
    assert (center - pre) / 0.1 == pytest.approx(R[:, 0])

    line = np.array([p.translation for p in poses["approach"]])
    assert len(line) == 5
    assert line[0] == pytest.approx(pre) and line[-1] == pytest.approx(center)
    assert np.diff(line, axis=0) == pytest.approx(np.tile((center - pre) / 4, (4, 1)))
    assert all(np.allclose(p.rotation, R) for p in poses["approach"])

    assert poses["lift"].translation == pytest.approx(center + [0, 0, 0.1])


def test_needs_two_waypoints():
    with pytest.raises(ValueError):
        grasp_poses(np.zeros(3), np.eye(3), n_waypoints=1)


START = {
    "lift_joint": 0.6,
    "arm_joint": 0.1,
    "wrist_yaw_joint": 0.0,
    "wrist_pitch_joint": 0.0,
    "wrist_roll_joint": 0.0,
}


@pytest.fixture(scope="module")
def urdf_xml():
    stretch4_urdf = pytest.importorskip("stretch4_urdf")
    return stretch4_urdf.get_urdf("SE4", "francis", "eoa_wrist_dw4_tool_sg4")


@pytest.fixture(scope="module")
def arm_solver(urdf_xml):
    from mm_ik.solver import IKSolver

    return IKSolver(urdf_xml, base_dofs=())


def reachable_center(arm_solver, arm=0.35, lift=0.8):
    return arm_solver.fk({**START, "arm_joint": arm, "lift_joint": lift}).translation


def test_plan_with_locked_base(arm_solver):
    R = fixed_grasp_rotation(arm_solver, {})
    center = reachable_center(arm_solver)
    plan = plan_grasp(arm_solver, START, center, R, n_waypoints=6)

    assert plan.base == {}
    assert len(plan.approach) == 5
    reached = [arm_solver.fk(j) for j in [plan.pregrasp] + plan.approach]
    for pose, target in zip(reached, plan.poses["approach"]):
        assert np.linalg.norm(pose.translation - target.translation) < 2e-3
        assert np.allclose(pose.rotation, R, atol=2e-2)
    assert arm_solver.fk(plan.lift).translation == pytest.approx(
        center + [0, 0, 0.1], abs=2e-3
    )


def test_plan_moves_base_only_once(urdf_xml, arm_solver):
    from mm_ik.solver import IKSolver

    solver = IKSolver(urdf_xml, base_dofs=("x",))
    R = fixed_grasp_rotation(arm_solver, {})
    center = reachable_center(arm_solver, arm=0.52) + [
        0.25,
        0,
        0,
    ]  # beyond the arm alone
    plan = plan_grasp(arm_solver, START, center, R, base_solver=solver)

    assert plan.base["base_x"] > 0.1
    moved = solver.base_motion({**plan.pregrasp, **plan.base})
    final = moved * arm_solver.fk(plan.approach[-1])
    assert np.linalg.norm(final.translation - center) < 2e-3
    assert all(set(j) == set(START) for j in plan.approach + [plan.lift])


def test_unreachable_raises(arm_solver):
    R = fixed_grasp_rotation(arm_solver, {})
    with pytest.raises(PlanningError, match="pre-grasp"):
        plan_grasp(arm_solver, START, [2.0, 0.0, 0.8], R)


def test_small_sideways_miss_is_accepted(arm_solver):
    """1.3 cm off the gripper's line: unreachable exactly, fine for grasping."""
    R = fixed_grasp_rotation(arm_solver, {})
    center = reachable_center(arm_solver) + [0, 0.013, 0]
    with pytest.raises(PlanningError):
        plan_grasp(arm_solver, START, center, R)
    plan = plan_grasp(
        arm_solver,
        START,
        center,
        R,
        accept_position_error=0.015,
        accept_orientation_error=np.deg2rad(3),
    )
    reached = arm_solver.fk(plan.approach[-1])
    assert np.linalg.norm(reached.translation - center) < 0.015


def test_free_yaw_centers_on_sideways_object(urdf_xml):
    """Object 2 cm off the gripper's line: free yaw ends exactly on it, approach straight along x."""
    from mm_ik.solver import IKSolver

    solver = IKSolver(
        urdf_xml, base_dofs=(), orientation_cost=FREE_YAW_ORIENTATION_COST
    )
    R = fixed_grasp_rotation(solver, {})
    center = reachable_center(solver) + [0, 0.02, 0]
    plan = plan_grasp(solver, START, center, R, n_waypoints=4)

    path = [solver.fk(j).translation for j in [plan.pregrasp] + plan.approach]
    assert np.linalg.norm(path[-1] - center) < 2e-3
    for p in path:  # straight line along base x: y and z stay on the object's
        assert p[1:] == pytest.approx(center[1:], abs=2e-3)
    yaws = [j["wrist_yaw_joint"] for j in [plan.pregrasp] + plan.approach]
    assert (
        max(yaws) - min(yaws) < 0.01 and yaws[0] > 0.05
    )  # turned left, constant on the way in


COMFORT = {"arm_joint": 0.35, "wrist_yaw_joint": 0.0}


def test_arm_first_base_untouched_when_reachable(urdf_xml):
    arm_solver, base_solver = make_solvers(urdf_xml, ("x", "y", "theta"))
    R = fixed_grasp_rotation(arm_solver, {})
    plan = plan_grasp(
        arm_solver,
        START,
        reachable_center(arm_solver),
        R,
        base_solver=base_solver,
        base_posture=COMFORT,
    )
    assert plan.base == {}


def test_base_translates_when_out_of_reach(urdf_xml):
    """25 cm past full extension and 15 cm to the left: base moves x and y, not theta."""
    arm_solver, base_solver = make_solvers(urdf_xml, ("x", "y", "theta"))
    R = fixed_grasp_rotation(arm_solver, {})
    center = reachable_center(arm_solver, arm=0.52) + [0.25, 0.15, 0.0]
    plan = plan_grasp(
        arm_solver, START, center, R, base_solver=base_solver, base_posture=COMFORT
    )

    assert plan.base["base_y"] == pytest.approx(0.15, abs=0.01)
    assert abs(plan.base["base_theta"]) < np.deg2rad(1)
    assert plan.pregrasp["arm_joint"] == pytest.approx(0.35, abs=0.03)
    moved = base_solver.base_motion({**plan.pregrasp, **plan.base})
    final = moved * arm_solver.fk(plan.approach[-1])
    assert np.linalg.norm(final.translation - center) < 2e-3


def test_far_sideways_needs_base_y(urdf_xml):
    """15 cm sideways: wrist yaw is capped at 10 deg (~5 cm), so x alone cannot do it."""
    R = None
    for dofs in (("x",), ("x", "y")):
        arm_solver, base_solver = make_solvers(urdf_xml, dofs)
        R = fixed_grasp_rotation(arm_solver, {})
        center = reachable_center(arm_solver) + [0.0, 0.15, 0.0]
        if dofs == ("x",):
            with pytest.raises(PlanningError):
                plan_grasp(
                    arm_solver,
                    START,
                    center,
                    R,
                    base_solver=base_solver,
                    base_posture=COMFORT,
                )
        else:
            plan = plan_grasp(
                arm_solver,
                START,
                center,
                R,
                base_solver=base_solver,
                base_posture=COMFORT,
            )
            assert plan.base["base_y"] == pytest.approx(0.15, abs=0.01)


def test_reach_uses_base_only_if_needed(urdf_xml):
    arm_solver, base_solver = make_solvers(urdf_xml, ("x", "y", "theta"))
    near = arm_solver.fk({**START, "arm_joint": 0.3})
    base, joints = reach(arm_solver, START, near, base_solver, COMFORT)
    assert base == {} and joints["arm_joint"] == pytest.approx(0.3, abs=1e-3)

    far = arm_solver.fk({**START, "arm_joint": 0.52})
    far.translation[0] += 0.3
    base, joints = reach(arm_solver, START, far, base_solver, COMFORT)
    moved = base_solver.base_motion({**joints, **base})
    assert (
        np.linalg.norm((moved * arm_solver.fk(joints)).translation - far.translation)
        < 2e-3
    )
    assert base["base_x"] > 0.25


def test_object_width_along_closing_axis():
    R = np.eye(3)  # closing axis = base y
    pts = np.column_stack(
        [np.linspace(0, 0.3, 200), np.linspace(-0.03, 0.03, 200), np.zeros(200)]
    )
    assert object_width(pts, R) == pytest.approx(0.06, abs=5e-3)


def test_moving_joints_leaves_out_still_joints():
    """Approach (arm only), then lift (lift only): each point carries just the moving joint."""
    start = {"lift_joint": 0.5, "arm_joint": 0.1, "wrist_yaw_joint": 0.0}
    min_change = {"lift_joint": 0.002, "arm_joint": 0.002, "wrist_yaw_joint": 0.005}
    points = [
        {"lift_joint": 0.5005, "arm_joint": 0.3, "wrist_yaw_joint": 0.001},
        {"lift_joint": 0.6, "arm_joint": 0.3, "wrist_yaw_joint": 0.001},
        {"lift_joint": 0.6, "arm_joint": 0.301, "wrist_yaw_joint": 0.0},
    ]
    assert moving_joints(start, points, min_change) == [
        {"arm_joint": 0.3},
        {"lift_joint": 0.6},
    ]


def test_moving_joints_accumulates_small_steps():
    """Steps below the threshold add up against the last commanded value, not the last point."""
    start = {"a": 0.0}
    points = [{"a": 0.0015}, {"a": 0.003}, {"a": 0.0045}]
    assert moving_joints(start, points, {"a": 0.002}) == [{"a": 0.003}]


def test_split_long_moves():
    start = {"lift": 0.5, "arm": 0.1}
    points = split_long_moves(
        start,
        [{"lift": 0.9, "arm": 0.1}, {"lift": 0.9, "arm": 0.15}],
        {"lift": 0.2, "arm": 0.15},
    )
    assert [round(p["lift"], 6) for p in points] == [0.7, 0.9, 0.9]
    assert points[-1] == {"lift": 0.9, "arm": 0.15}
    prev = start
    for p in points:
        assert (
            abs(p["lift"] - prev["lift"]) <= 0.2 + 1e-9
            and abs(p["arm"] - prev["arm"]) <= 0.15 + 1e-9
        )
        prev = p


def test_split_exact_multiple_does_not_add_a_point():
    assert len(split_long_moves({"a": 0.0}, [{"a": 0.4}], {"a": 0.2})) == 2
