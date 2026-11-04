import pytest

from mm_ik.solver import ARM_JOINT_NAMES
from mm_manip.motion.velocity import BASE_X, PathTracker, TrackingGains

GAINS = TrackingGains(
    max_velocity={"lift": 0.1, "yaw": 1.0},
    tolerance={"lift": 0.003, "yaw": 0.01},
    gain=2.0,
    min_segment=0.3,
)


def test_segments_are_timed_by_the_slowest_joint():
    tracker = PathTracker({"lift": 0.0, "yaw": 0.0}, [{"lift": 0.2, "yaw": 0.5}], GAINS)
    assert tracker.duration == pytest.approx(2.0)  # lift: 0.2 m at 0.1 m/s
    position, velocity = tracker.desired(1.0)
    assert position == pytest.approx({"lift": 0.1, "yaw": 0.25})
    assert velocity == pytest.approx({"lift": 0.1, "yaw": 0.25})


def test_short_segments_take_min_segment():
    tracker = PathTracker(
        {"lift": 0.0, "yaw": 0.0}, [{"lift": 0.001, "yaw": 0.0}], GAINS
    )
    assert tracker.duration == pytest.approx(0.3)


def test_no_stop_at_intermediate_waypoints():
    points = [{"lift": 0.1, "yaw": 0.0}, {"lift": 0.2, "yaw": 0.0}]
    tracker = PathTracker({"lift": 0.0, "yaw": 0.0}, points, GAINS)
    _, velocity = tracker.desired(1.0 - 1e-6)
    _, after = tracker.desired(1.0 + 1e-6)
    assert velocity["lift"] == pytest.approx(0.1) and after["lift"] == pytest.approx(
        0.1
    )


def test_command_is_path_velocity_plus_correction():
    tracker = PathTracker({"lift": 0.0, "yaw": 0.0}, [{"lift": 0.2, "yaw": 0.0}], GAINS)
    command, error, done = tracker.command(1.0, {"lift": 0.09, "yaw": 0.0})
    assert error["lift"] == pytest.approx(0.01)
    assert command["lift"] == pytest.approx(0.1)  # 0.1 + 2 * 0.01, clamped to 0.1
    command, _, _ = tracker.command(1.0, {"lift": 0.11, "yaw": 0.0})
    assert command["lift"] == pytest.approx(0.08)
    assert not done


def test_done_only_after_the_path_ends_and_within_tolerance():
    tracker = PathTracker({"lift": 0.0, "yaw": 0.0}, [{"lift": 0.2, "yaw": 0.0}], GAINS)
    at_end = {"lift": 0.2, "yaw": 0.0}
    assert not tracker.command(1.0, at_end)[2]  # path not over yet
    assert tracker.command(2.0, at_end)[2]
    assert not tracker.command(2.0, {"lift": 0.19, "yaw": 0.0})[2]  # 1 cm short


def test_holds_the_end_after_the_path():
    tracker = PathTracker({"lift": 0.0, "yaw": 0.0}, [{"lift": 0.2, "yaw": 0.0}], GAINS)
    command, _, _ = tracker.command(5.0, {"lift": 0.21, "yaw": 0.0})
    assert command["lift"] == pytest.approx(-0.02)  # back toward 0.2, no feedforward


def test_only_the_joints_in_the_points():
    tracker = PathTracker(
        {"lift": 0.0, "yaw": 0.3, "extra": 1.0}, [{"lift": 0.1}], GAINS
    )
    assert tracker.joints == ["lift"]


def test_needs_a_waypoint():
    with pytest.raises(ValueError):
        PathTracker({"lift": 0.0}, [], GAINS)


def test_default_gains_cover_the_arm_and_the_base():
    gains = TrackingGains()
    for joint in [*ARM_JOINT_NAMES, BASE_X]:
        assert joint in gains.max_velocity and joint in gains.tolerance
