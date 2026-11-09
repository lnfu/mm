"""Move the base on a velocity path, then print the odometry. See NOTE.md."""

import argparse
import math
import sys
import time

from rclpy.utilities import remove_ros_args

from ..cancel import CancelToken
from ..config import ArticulateConfig
from ..motion.base_control import displacement
from ..motion.velocity import BASE_THETA, BASE_X, BASE_Y, TrackingGains
from .common import run_tool, wait_for


def run(args, robot, cfg):
    arm = wait_for(robot.arm_joints, "/joint_states")
    start = wait_for(lambda: robot.odom_pose(max_age=0.5), "/wheel_odom")
    theta = math.radians(args.theta_deg)
    print(
        f"moving the base x {args.x:+.3f} m, y {args.y:+.3f} m, "
        f"theta {args.theta_deg:+.1f} deg; mark the floor now (Ctrl-C to cancel)"
    )
    time.sleep(args.delay)
    gains = TrackingGains(
        max_velocity=cfg.path_max_velocity,
        tolerance=cfg.path_tolerance,
        gain=cfg.tracking_gain,
        min_segment=cfg.path_min_segment,
    )
    target = {**arm, BASE_X: args.x, BASE_Y: args.y, BASE_THETA: theta}
    robot.follow_path([target], CancelToken(), gains)
    time.sleep(1.0)  # let /wheel_odom catch up
    dx, dy, dyaw = displacement(start, robot.odom_pose(max_age=0.5))
    print(
        f"/wheel_odom says: x {dx:+.3f} m, y {dy:+.3f} m, theta "
        f"{math.degrees(dyaw):+.1f} deg (asked {args.x:+.3f}, {args.y:+.3f}, "
        f"{args.theta_deg:+.1f})"
    )
    print("now measure the real displacement against your floor marks")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Move the base, print odometry.")
    parser.add_argument("x", type=float, help="m forward")
    parser.add_argument("y", type=float, help="m to the left")
    parser.add_argument("--theta-deg", type=float, default=0.0, help="turn, deg")
    parser.add_argument(
        "--delay", type=float, default=3.0, help="s before moving, to mark the floor"
    )
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    return run_tool(
        lambda node, runtime, shared, cfg: run(args, runtime.robot, cfg),
        ArticulateConfig,
    )


if __name__ == "__main__":
    sys.exit(main())
