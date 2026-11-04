"""Hardware interface to stretch_driver."""

import threading
import time

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from mm_ik.model import ARM_SEGMENTS
from mm_ik.solver import ARM_JOINT_NAMES

from ..errors import Canceled
from .base_control import (
    BaseGains,
    Pose2D,
    rotate_command,
    translate_command,
    yaw_from_quaternion,
)
from .trajectory import (
    DEFAULT_MAX_STEP,
    MIN_JOINT_CHANGE,
    joint_trajectory,
    moving_joints,
    split_long_moves,
)


class MotionError(RuntimeError):
    pass


class Robot:
    def __init__(
        self,
        node,
        fjt_action,
        driver_joint_names: dict,
        max_step: dict = DEFAULT_MAX_STEP,
        no_split_action_timeout: float = 10.0,
        callback_group=None,
    ):
        """
        Args:
            fjt_action: FollowJointTrajectory action name.
            driver_joint_names: IK joint name (ARM_JOINT_NAMES) -> name the driver uses
                in joint_states and FollowJointTrajectory.
            max_step: largest change per joint in one trajectory point, used when the
                driver's action_timeout is shorter than no_split_action_timeout.
        """
        self._node = node
        self._driver_names = driver_joint_names
        self._max_step = max_step
        self._no_split_action_timeout = no_split_action_timeout
        self._split = None  # last answer of _needs_split, to log only changes
        self._lock = threading.Lock()
        self._joint_state = {}
        self._urdf = None
        self._active_goal = None

        node.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_states,
            10,
            callback_group=callback_group,
        )
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(
            String,
            "/robot_description",
            self._on_urdf,
            latched,
            callback_group=callback_group,
        )
        self._fjt = ActionClient(
            node, FollowJointTrajectory, fjt_action, callback_group=callback_group
        )
        self._driver_params = node.create_client(
            GetParameters,
            "/stretch_driver/get_parameters",
            callback_group=callback_group,
        )

        # Base: stretch_driver takes /cmd_vel (vx, vy, w) in navigation mode and has no
        # cmd_vel timeout, so every base motion ends by publishing zero.
        self._odom = None  # (Pose2D, receive time)
        node.create_subscription(
            Odometry, "/wheel_odom", self._on_odom, 10, callback_group=callback_group
        )
        self._cmd_vel = node.create_publisher(Twist, "/cmd_vel", 1)
        self._base_moving = False

    # --- state ------------------------------------------------------------

    def _on_joint_states(self, msg):
        with self._lock:
            self._joint_state.update(zip(msg.name, msg.position))

    def _on_odom(self, msg):
        p = msg.pose.pose
        pose = Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        with self._lock:
            self._odom = (pose, time.monotonic())

    def _on_urdf(self, msg):
        self._urdf = msg.data

    @property
    def urdf(self):
        return self._urdf

    def arm_joints(self) -> dict:
        """Current arm joints keyed by IK names.

        The arm is read from its driver name if published, otherwise summed from
        the URDF segments arm_l1..arm_l4 (stretch_driver publishes the segments,
        each at a quarter of the total).
        """
        with self._lock:
            state = dict(self._joint_state)
        joints = {}
        for name in ARM_JOINT_NAMES:
            driver = self._driver_names[name]
            if driver in state:
                joints[name] = state[driver]
            elif name == "arm_joint" and all(s in state for s in ARM_SEGMENTS):
                joints[name] = sum(state[s] for s in ARM_SEGMENTS)
            else:
                raise MotionError(f"{driver} not in /joint_states")
        return joints

    # --- arm --------------------------------------------------------------

    def move_arm(self, start: dict, points: list, should_cancel):
        """Move from `start` through `points` (IK-name joint dicts) and wait.

        stretch_driver runs a trajectory point by point anyway (no interpolation,
        time_from_start ignored), so each point goes out as its own goal with only the
        joints that move (see trajectory.moving_joints). The driver aborts a point after its
        action_timeout, so long moves are cut into max_step pieces when it is short.
        Raises Canceled if should_cancel() turns true (the goal is canceled first),
        MotionError if the controller rejects or fails it.
        """
        if not self._fjt.wait_for_server(timeout_sec=2.0):
            raise MotionError("FollowJointTrajectory server not available")
        if self._needs_split(should_cancel):
            points = split_long_moves(start, points, self._max_step)
        for point in moving_joints(start, points, MIN_JOINT_CHANGE):
            self._move_to(point, should_cancel)

    def _needs_split(self, should_cancel, timeout=2.0) -> bool:
        """Whether stretch_driver's action_timeout (3 s by default) is too short for
        long moves. Asked on every move, since `ros2 param set /stretch_driver
        action_timeout 15.0` changes it at run time. Unknown counts as too short."""
        action_timeout = None
        if self._driver_params.wait_for_service(timeout_sec=timeout):
            future = self._driver_params.call_async(
                GetParameters.Request(names=["action_timeout"])
            )
            deadline = time.monotonic() + timeout
            while not future.done() and time.monotonic() < deadline:
                if should_cancel():
                    raise Canceled()
                time.sleep(0.02)
            if future.done() and future.result().values:
                action_timeout = future.result().values[0].double_value
        split = action_timeout is None or action_timeout < self._no_split_action_timeout
        if split != self._split:
            self._split = split
            self._node.get_logger().info(
                f"stretch_driver action_timeout {action_timeout} s: "
                f"{'splitting long moves' if split else 'one point per move'}"
            )
        return split

    def _move_to(self, point: dict, should_cancel):
        names = list(point)
        traj = joint_trajectory(
            [self._driver_names[n] for n in names],
            [{self._driver_names[n]: point[n] for n in names}],
            [1.0],
        )
        goal_handle = self._wait(
            self._fjt.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj)),
            should_cancel,
        )
        if not goal_handle.accepted:
            raise MotionError("trajectory rejected")
        self._active_goal = goal_handle
        try:
            result = self._wait(goal_handle.get_result_async(), should_cancel).result
        finally:
            self._active_goal = None
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise MotionError(
                f"trajectory failed ({result.error_code}): {result.error_string}"
            )

    def _wait(self, future, should_cancel):
        while not future.done():
            if should_cancel():
                self.stop()
                raise Canceled()
            time.sleep(0.02)
        return future.result()

    def stop(self):
        """Stop all motion now."""
        goal = self._active_goal
        if goal is not None:
            goal.cancel_goal_async()
        if self._base_moving:
            self._publish_twist(0.0, 0.0, 0.0)

    # --- base -------------------------------------------------------------

    def move_base(
        self,
        base_x: float,
        base_y: float,
        base_theta: float,
        should_cancel,
        gains: BaseGains = None,
        rate_hz: float = 20.0,
    ):
        """Move the base by (x, y, theta) in the starting base_link and wait.

        Translates by (x, y), then turns in place by theta, closed loop on /wheel_odom
        (the IK's base chain is x, y, then theta). Always ends by publishing zero.
        Raises Canceled or MotionError (no odometry, odometry stalled, or timeout).
        """
        gains = gains or BaseGains()
        start = self._odom_pose(max_age=0.5)
        distance = (base_x**2 + base_y**2) ** 0.5
        timeout = 5.0 + 2.0 * (
            distance / gains.max_linear + abs(base_theta) / gains.max_angular
        )
        deadline = time.monotonic() + timeout
        self._base_moving = True
        try:
            for command in (
                lambda p: translate_command(start, p, base_x, base_y, gains),
                lambda p: rotate_command(start, p, base_theta, gains),
            ):
                while True:
                    if should_cancel():
                        raise Canceled()
                    if time.monotonic() > deadline:
                        raise MotionError(
                            f"base motion timed out after {timeout:.0f} s"
                        )
                    vx, vy, w, done = command(self._odom_pose(max_age=0.5))
                    if done:
                        break
                    self._publish_twist(vx, vy, w)
                    time.sleep(1.0 / rate_hz)
                self._publish_twist(0.0, 0.0, 0.0)
        finally:
            self._publish_twist(0.0, 0.0, 0.0)
            self._base_moving = False

    def _odom_pose(self, max_age: float) -> Pose2D:
        with self._lock:
            odom = self._odom
        if odom is None:
            raise MotionError("no /wheel_odom received")
        pose, stamp = odom
        if time.monotonic() - stamp > max_age:
            raise MotionError(f"/wheel_odom is older than {max_age} s")
        return pose

    def _publish_twist(self, vx, vy, w):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(w)
        self._cmd_vel.publish(msg)

    # --- gripper ----------------------------------------------------------
    # TODO[lnfu] 夾爪壞了...

    def open_gripper(self, should_cancel):
        raise NotImplementedError("gripper (mechanism being repaired)")

    def close_gripper(self, should_cancel):
        raise NotImplementedError("gripper (mechanism being repaired)")

    def gripper_position(self) -> float:
        raise NotImplementedError("gripper (mechanism being repaired)")
