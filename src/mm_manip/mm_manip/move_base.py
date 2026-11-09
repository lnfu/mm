"""Measure base odometry: move the base as articulate does, then measure by hand.

Stop manipulation_server first (this node takes its name to read its parameters):

  python3 -m mm_manip.move_base 0.0 0.10 \\
      --ros-args --params-file src/mm_manip/config/manipulation.yaml
  python3 -m mm_manip.move_base 0.137 0.099 --theta-deg 0 --ros-args ...

x, y (m) are along base_link's x (forward) and y (left) where the base starts.
The base moves on a velocity path (Robot.follow_path, closed loop on /wheel_odom,
articulate's path gains and tolerances) with the arm held where it is, then the
displacement /wheel_odom saw is printed: compare it with a tape measure.
"""

import argparse
import math
import sys
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.utilities import remove_ros_args

from .config import ArticulateConfig, ServerConfig
from .motion.base_control import displacement
from .motion.robot import MotionError, Robot
from .motion.velocity import BASE_THETA, BASE_X, BASE_Y, TrackingGains


def _wait_for(get, timeout, what):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            return get()
        except MotionError:
            time.sleep(0.05)
    raise RuntimeError(f"no {what} after {timeout} s (is stretch_driver running?)")


def run(args, robot, cfg):
    arm = _wait_for(robot.arm_joints, 5.0, "/joint_states")
    start = _wait_for(lambda: robot._odom_pose(max_age=0.5), 5.0, "/wheel_odom")
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
    robot.follow_path([target], lambda: False, gains)
    time.sleep(1.0)  # let the base stop and /wheel_odom catch up
    dx, dy, dyaw = displacement(start, robot._odom_pose(max_age=0.5))
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

    rclpy.init()
    node = rclpy.create_node("mm_manip")
    ServerConfig.declare(node)
    ArticulateConfig.declare(node)
    shared, cfg = ServerConfig.from_node(node), ArticulateConfig.from_node(node)
    robot = Robot(
        node,
        shared.fjt_action,
        shared.driver_joint,
        control_rate=shared.control_rate,
        path_settle_timeout=shared.path_settle_timeout,
        callback_group=ReentrantCallbackGroup(),
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        return run(args, robot, cfg)
    except KeyboardInterrupt:
        robot.stop()
        print("interrupted: motion canceled")
        return 1
    except (MotionError, RuntimeError) as e:
        robot.stop()
        print(f"error: {e}")
        return 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
