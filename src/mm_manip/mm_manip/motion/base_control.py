"""Closed-loop base motion on wheel odometry. See NOTE.md."""

import math
from dataclasses import dataclass


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class BaseGains:
    max_linear: float = 0.05  # m/s
    min_linear: float = 0.01  # m/s, against friction at the end
    linear_gain: float = 1.0  # 1/s
    linear_tolerance: float = 0.01  # m
    max_angular: float = 0.2  # rad/s
    min_angular: float = 0.03  # rad/s
    angular_gain: float = 1.5  # 1/s
    angular_tolerance: float = math.radians(1.0)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def displacement(start: Pose2D, current: Pose2D):
    """(dx, dy, dyaw) travelled since start, in the starting base frame."""
    c, s = math.cos(start.yaw), math.sin(start.yaw)
    ex, ey = current.x - start.x, current.y - start.y
    return c * ex + s * ey, -s * ex + c * ey, wrap(current.yaw - start.yaw)


def _speed(error, gain, lo, hi):
    return min(max(gain * error, lo), hi)


def translate_command(
    start: Pose2D, current: Pose2D, goal_x: float, goal_y: float, g: BaseGains
):
    """(vx, vy, w, done): go to (goal_x, goal_y) in the start frame, same heading."""
    dx, dy, dyaw = displacement(start, current)
    ex, ey = goal_x - dx, goal_y - dy
    dist = math.hypot(ex, ey)
    if dist < g.linear_tolerance:
        return 0.0, 0.0, 0.0, True
    speed = _speed(dist, g.linear_gain, g.min_linear, g.max_linear)
    sx, sy = ex / dist * speed, ey / dist * speed  # in the start frame
    c, s = math.cos(dyaw), math.sin(dyaw)  # into the current robot frame
    vx, vy = c * sx + s * sy, -s * sx + c * sy
    w = max(min(-g.angular_gain * dyaw, g.max_angular), -g.max_angular)
    return vx, vy, w, False


def rotate_command(start: Pose2D, current: Pose2D, goal_theta: float, g: BaseGains):
    """(0, 0, w, done): turn in place to goal_theta from the start heading."""
    err = wrap(goal_theta - displacement(start, current)[2])
    if abs(err) < g.angular_tolerance:
        return 0.0, 0.0, 0.0, True
    w = math.copysign(
        _speed(abs(err), g.angular_gain, g.min_angular, g.max_angular), err
    )
    return 0.0, 0.0, w, False


def yaw_from_quaternion(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
