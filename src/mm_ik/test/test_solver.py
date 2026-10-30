"""FK -> IK round-trip on the real Stretch 4 URDF (from hello-robot-stretch4-urdf)."""

import numpy as np
import pinocchio as pin
import pytest

stretch4_urdf = pytest.importorskip('stretch4_urdf')

from mm_ik.solver import IKSolver  # noqa: E402

START = {
    'lift_joint': 0.6,
    'arm_joint': 0.1,
    'wrist_yaw_joint': 0.0,
    'wrist_pitch_joint': 0.0,
    'wrist_roll_joint': 0.0,
}


@pytest.fixture(scope='module')
def urdf_xml():
    return stretch4_urdf.get_urdf('SE4', 'francis', 'eoa_wrist_dw4_tool_sg4')


def random_joints(solver, rng):
    lo, hi = solver.model.lowerPositionLimit, solver.model.upperPositionLimit
    # Stay away from limits and from wrist singularities.
    q = lo + (hi - lo) * rng.uniform(0.2, 0.8, size=lo.shape)
    joints = solver.to_joints(q)
    for wrist in ('wrist_yaw_joint', 'wrist_pitch_joint', 'wrist_roll_joint'):
        joints[wrist] = rng.uniform(-0.6, 0.6)
    for base in ('base_x', 'base_y', 'base_theta'):
        if base in joints:
            joints[base] = rng.uniform(-0.3, 0.3)
    return joints


@pytest.mark.parametrize('base_dofs', [(), ('x',), ('x', 'y', 'theta')])
def test_round_trip(urdf_xml, base_dofs):
    solver = IKSolver(urdf_xml, base_dofs=base_dofs)
    rng = np.random.default_rng(0)
    for _ in range(10):
        goal = random_joints(solver, rng)
        target = solver.fk(goal)
        result = solver.solve(START, target)
        assert result.converged, (goal, result)
        reached = solver.fk(result.joints)
        assert np.linalg.norm(reached.translation - target.translation) < 2e-3
        assert np.linalg.norm(pin.log3(reached.rotation.T @ target.rotation)) < 2e-2
        lo, hi = solver.model.lowerPositionLimit, solver.model.upperPositionLimit
        q = solver.to_q(result.joints)
        assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6)


def test_locked_base_does_not_move(urdf_xml):
    solver = IKSolver(urdf_xml, base_dofs=())
    assert not any(n.startswith('base_') for n in solver.joint_names)


def test_base_helps_reach_beyond_arm(urdf_xml):
    """A target 30 cm past full arm extension needs base_x."""
    far = IKSolver(urdf_xml, base_dofs=()).fk({**START, 'arm_joint': 0.52})
    far.translation[0] += 0.3

    assert not IKSolver(urdf_xml, base_dofs=()).solve(START, far).converged
    result = IKSolver(urdf_xml, base_dofs=('x',)).solve(START, far)
    assert result.converged
    assert result.joints['base_x'] > 0.25


def test_solve_path_straight_line(urdf_xml):
    solver = IKSolver(urdf_xml, base_dofs=('x',))
    goal = solver.fk({**START, 'arm_joint': 0.4})
    pre = pin.SE3(goal.rotation, goal.translation - goal.rotation[:, 0] * 0.1)
    line = [pin.SE3(goal.rotation, pre.translation + s * (goal.translation - pre.translation))
            for s in np.linspace(0, 1, 6)]
    results = solver.solve_path(START, line)
    assert len(results) == len(line) and all(r.converged for r in results)
    for r, target in zip(results, line):
        assert np.linalg.norm(solver.fk(r.joints).translation - target.translation) < 2e-3


def test_base_motion(urdf_xml):
    solver = IKSolver(urdf_xml, base_dofs=('x', 'y', 'theta'))
    assert np.allclose(solver.base_motion(START).homogeneous, np.eye(4))
    moved = solver.base_motion({**START, 'base_x': 0.2, 'base_y': -0.1, 'base_theta': 0.5})
    assert moved.translation[:2] == pytest.approx([0.2, -0.1], abs=1e-6)
    assert np.arctan2(moved.rotation[1, 0], moved.rotation[0, 0]) == pytest.approx(0.5)
    # The ee seen from the moved base equals the arm-only FK.
    arm_only = IKSolver(urdf_xml, base_dofs=())
    joints = {**START, 'base_x': 0.2, 'base_y': -0.1, 'base_theta': 0.5}
    assert np.allclose((moved.actInv(solver.fk(joints))).homogeneous, arm_only.fk(START).homogeneous)


def test_free_yaw_reaches_sideways_target(urdf_xml):
    """With yaw free, a target 3 cm off the gripper's line is reached exactly, gripper still level."""
    fixed = IKSolver(urdf_xml, base_dofs=())
    free = IKSolver(urdf_xml, base_dofs=(), orientation_cost=(0.5, 0.5, 0.0))
    target = fixed.fk({**START, 'arm_joint': 0.3})
    target.translation[1] += 0.03

    assert not fixed.solve(START, target).converged
    result = free.solve(START, target)
    assert result.converged
    reached = free.fk(result.joints)
    assert np.linalg.norm(reached.translation - target.translation) < 2e-3
    assert abs(result.joints['wrist_pitch_joint']) < 1e-2 and abs(result.joints['wrist_roll_joint']) < 1e-2
    assert result.joints['wrist_yaw_joint'] == pytest.approx(np.arcsin(0.03 / 0.308), abs=0.02)


def test_base_does_the_travelling_with_comfort_posture(urdf_xml):
    """Cheap base, dearer arm, arm biased to 0.35: base translates, no rotation, arm near 0.35."""
    solver = IKSolver(urdf_xml, base_dofs=('x', 'y', 'theta'), orientation_cost=(0.5, 0.5, 0.0),
                      posture_cost=1e-2, posture_costs={'base_x': 1e-4, 'base_y': 1e-4, 'base_theta': 1e-1})
    far = IKSolver(urdf_xml, base_dofs=()).fk({**START, 'arm_joint': 0.52})
    far.translation += [0.20, 0.15, 0.0]
    result = solver.solve(START, far, posture={'arm_joint': 0.35, 'wrist_yaw_joint': 0.0})
    assert result.converged
    j = result.joints
    assert j['arm_joint'] == pytest.approx(0.35, abs=0.03)
    assert j['base_y'] == pytest.approx(0.15, abs=0.01)
    assert abs(j['base_theta']) < np.deg2rad(1) and abs(j['wrist_yaw_joint']) < np.deg2rad(1)
    assert np.linalg.norm(solver.fk(j).translation - far.translation) < 2e-3


def test_base_stage_settles_on_posture_not_first_reach(urdf_xml):
    """A 23 cm sideways gap: the base covers it, not the wrist yaw, even though the
    per-iteration velocity limit lets yaw reach the target first."""
    solver = IKSolver(urdf_xml, base_dofs=('x', 'y'), orientation_cost=(0.5, 0.5, 0.0),
                      joint_limits={'wrist_yaw_joint': (-0.17, 0.17)},
                      posture_cost=1e-2, posture_costs={'base_x': 1e-4, 'base_y': 1e-4})
    target = IKSolver(urdf_xml, base_dofs=()).fk({**START, 'arm_joint': 0.35})
    target.translation[1] += 0.23
    j = solver.solve(START, target, posture={'arm_joint': 0.35, 'wrist_yaw_joint': 0.0}).joints
    assert j['base_y'] == pytest.approx(0.23, abs=0.01)
    assert abs(j['wrist_yaw_joint']) < np.deg2rad(1)
    assert j['arm_joint'] == pytest.approx(0.35, abs=0.02)


def test_unknown_base_dof(urdf_xml):
    with pytest.raises(ValueError):
        IKSolver(urdf_xml, base_dofs=('z',))
