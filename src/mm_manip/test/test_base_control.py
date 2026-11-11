import math
from types import SimpleNamespace as NS

import pytest

from mm_manip.motion.base_control import (
    BaseGains,
    Pose2D,
    displacement,
    rotate_command,
    translate_command,
    wrap,
    yaw_from_quaternion,
)


def simulate(start, command, steps=2000, dt=0.05, slip=1.0):
    """Integrate (vx, vy, w) in the robot frame until done; slip < 1 makes the robot fall short."""
    p = Pose2D(start.x, start.y, start.yaw)
    for _ in range(steps):
        vx, vy, w, done = command(p)
        if done:
            return p
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        p = Pose2D(
            p.x + (c * vx - s * vy) * dt * slip,
            p.y + (s * vx + c * vy) * dt * slip,
            wrap(p.yaw + w * dt * slip),
        )
    raise AssertionError("did not finish")


@pytest.mark.parametrize("start_yaw", [0.0, 1.0, -2.5])
@pytest.mark.parametrize("slip", [1.0, 0.7])
def test_translate_reaches_goal_in_start_frame(start_yaw, slip):
    g = BaseGains()
    start = Pose2D(1.0, -2.0, start_yaw)
    end = simulate(
        start, lambda p: translate_command(start, p, 0.3, -0.15, g), slip=slip
    )
    dx, dy, dyaw = displacement(start, end)
    assert math.hypot(dx - 0.3, dy + 0.15) < g.linear_tolerance
    assert abs(dyaw) < math.radians(2)


def test_translate_never_exceeds_max_speed():
    g = BaseGains()
    vx, vy, _, done = translate_command(Pose2D(0, 0, 0), Pose2D(0, 0, 0), 1.0, 1.0, g)
    assert not done and math.hypot(vx, vy) == pytest.approx(g.max_linear)


@pytest.mark.parametrize("goal", [0.5, -0.5, math.radians(170)])
def test_rotate_reaches_goal(goal):
    g = BaseGains()
    start = Pose2D(0.2, 0.1, 3.0)  # near +pi, so the heading wraps on the way
    end = simulate(start, lambda p: rotate_command(start, p, goal, g))
    assert abs(wrap(displacement(start, end)[2] - goal)) < g.angular_tolerance


def test_yaw_from_quaternion():
    s = math.sqrt(0.5)
    assert yaw_from_quaternion(NS(x=0, y=0, z=s, w=s)) == pytest.approx(math.pi / 2)
