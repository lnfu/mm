import pytest

from mm_grasp.motion.trajectory import (
    moving_joints,
    split_long_moves,
    urdf_joint_positions,
)


def test_urdf_joint_positions_splits_arm():
    names, positions = urdf_joint_positions({"lift_joint": 0.6, "arm_joint": 0.4})
    assert names == [
        "lift_joint",
        "arm_l1_joint",
        "arm_l2_joint",
        "arm_l3_joint",
        "arm_l4_joint",
    ]
    assert positions == pytest.approx([0.6, 0.1, 0.1, 0.1, 0.1])


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
