from mm_manip.motion.gripper import GripperSettings, StallDetector

G = GripperSettings(still=0.005, stall_time=0.3)


def test_fingers_that_move_then_stop_are_stopped():
    stall = StallDetector(G, 1.0, now=0.0)
    assert not stall.stopped(0.9, 0.1)
    assert not stall.stopped(0.8, 0.2)
    assert not stall.stopped(0.8, 0.4)
    assert stall.stopped(0.801, 0.55)


def test_small_changes_count_as_still():
    stall = StallDetector(G, 1.0, now=0.0)
    assert not stall.stopped(0.9, 0.1)
    assert stall.stopped(0.903, 0.45)


def test_fingers_that_never_move_are_stopped_after_one_second():
    stall = StallDetector(G, 1.0, now=0.0)
    assert not stall.stopped(1.0, 0.5)
    assert not stall.stopped(1.0, 0.9)
    assert stall.stopped(1.0, 1.1)
