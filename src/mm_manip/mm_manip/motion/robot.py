import math
import threading
import time
from dataclasses import dataclass

from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointJog
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from mm_ik.model import ARM_SEGMENTS
from mm_ik.solver import ARM_JOINT_NAMES

from ..cancel import POLL, CancelToken
from ..errors import Canceled, SkillError
from .base_control import (
    BaseGains,
    Pose2D,
    displacement,
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
from .velocity import (
    BASE_JOINTS,
    BASE_THETA,
    BASE_X,
    BASE_Y,
    PathTracker,
    TrackingGains,
)

# /joint_vel: lift and arm get a velocity (m/s). The wrist joints get a
# distance (rad), because the driver does move_by(value * duration).
PRISMATIC = ("lift_joint", "arm_joint")


class MotionError(SkillError):
    pass


@dataclass(frozen=True)
class GripperSettings:
    joint: str = "gripper_joint"
    state_joint: str = "gripper_finger_left_joint"
    open: float = 80.0  # driver units, past the mechanical stop
    closed: float = -10.0  # driver units, past the mechanical stop
    closed_finger: float = 0.0  # rad, finger reading when closed on nothing
    holding_margin: float = 0.05  # rad
    still: float = 0.005  # rad
    stall_time: float = 0.3  # s
    timeout: float = 5.0  # s


class Robot:
    def __init__(
        self,
        node,
        fjt_action,
        driver_joint_names: dict,
        max_step: dict = DEFAULT_MAX_STEP,
        no_split_action_timeout: float = 10.0,
        fjt_mode: str = "navigation",
        control_rate: float = 30.0,
        path_settle_timeout: float = 5.0,
        gripper: "GripperSettings" = None,
        callback_group=None,
    ):
        self._node = node
        self._driver_names = driver_joint_names
        self._max_step = max_step
        self._no_split_action_timeout = no_split_action_timeout
        self._split = None
        self._fjt_mode = fjt_mode
        self._period = 1.0 / control_rate
        self._path_settle_timeout = path_settle_timeout
        self._gripper = gripper or GripperSettings()
        self._lock = threading.Lock()
        self._joint_state = {}
        self._joint_state_time = None
        self._mode = None
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
        self._driver_set_params = node.create_client(
            SetParameters,
            "/stretch_driver/set_parameters",
            callback_group=callback_group,
        )
        node.create_subscription(
            String, "/mode", self._on_mode, latched, callback_group=callback_group
        )
        self._joint_vel = node.create_publisher(JointJog, "/joint_vel", 1)
        self._arm_moving = False

        self._odom = None
        node.create_subscription(
            Odometry, "/wheel_odom", self._on_odom, 10, callback_group=callback_group
        )
        # stretch_driver has no /cmd_vel timeout. Always send zero to stop the base.
        self._cmd_vel = node.create_publisher(Twist, "/cmd_vel", 1)
        self._base_moving = False

    def _on_joint_states(self, msg):
        with self._lock:
            self._joint_state.update(zip(msg.name, msg.position, strict=False))
            self._joint_state_time = time.monotonic()

    def _on_odom(self, msg):
        p = msg.pose.pose
        pose = Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        with self._lock:
            self._odom = (pose, time.monotonic())

    def _on_urdf(self, msg):
        self._urdf = msg.data

    def _on_mode(self, msg):
        self._mode = msg.data

    @property
    def urdf(self):
        return self._urdf

    def arm_joints(self, max_age: float = None) -> dict:
        with self._lock:
            state = dict(self._joint_state)
            stamp = self._joint_state_time
        if max_age is not None and (
            stamp is None or time.monotonic() - stamp > max_age
        ):
            raise MotionError(f"/joint_states is older than {max_age} s")
        joints = {}
        for name in ARM_JOINT_NAMES:
            driver = self._driver_names[name]
            if driver in state:
                joints[name] = state[driver]
            # stretch_driver publishes arm_l1..arm_l4, each 1/4 of arm_joint.
            elif name == "arm_joint" and all(s in state for s in ARM_SEGMENTS):
                joints[name] = sum(state[s] for s in ARM_SEGMENTS)
            else:
                raise MotionError(f"{driver} not in /joint_states")
        return joints

    def move_arm(self, start: dict, points: list, cancel: CancelToken):
        self._ensure_mode(self._fjt_mode, cancel)
        if not self._fjt.wait_for_server(timeout_sec=2.0):
            raise MotionError("FollowJointTrajectory server not available")
        if self._needs_split(cancel):
            points = split_long_moves(start, points, self._max_step)
        # stretch_driver stops at each point and ignores time_from_start.
        # A lift or arm target at the current position never completes.
        # Thus, send only the joints that move.
        for point in moving_joints(start, points, MIN_JOINT_CHANGE):
            self._move_to(point, cancel)

    def _needs_split(self, cancel: CancelToken, timeout=2.0) -> bool:
        # stretch_driver aborts a point after action_timeout (default 3 s).
        # Read it on each move, because it can change at run time.
        action_timeout = None
        if self._driver_params.wait_for_service(timeout_sec=timeout):
            future = self._driver_params.call_async(
                GetParameters.Request(names=["action_timeout"])
            )
            if cancel.wait_until(future.done, timeout) and future.result().values:
                action_timeout = future.result().values[0].double_value
        split = action_timeout is None or action_timeout < self._no_split_action_timeout
        if split != self._split:
            self._split = split
            self._node.get_logger().info(
                f"stretch_driver action_timeout {action_timeout} s: "
                f"{'splitting long moves' if split else 'one point per move'}"
            )
        return split

    def _move_to(self, point: dict, cancel: CancelToken):
        names = list(point)
        traj = joint_trajectory(
            [self._driver_names[n] for n in names],
            [{self._driver_names[n]: point[n] for n in names}],
            [1.0],
        )
        goal_handle = cancel.result(
            self._fjt.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj)),
            on_cancel=self.stop,
        )
        if not goal_handle.accepted:
            raise MotionError("trajectory rejected")
        self._active_goal = goal_handle
        try:
            result = cancel.result(
                goal_handle.get_result_async(), on_cancel=self.stop
            ).result
        finally:
            self._active_goal = None
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise MotionError(
                f"trajectory failed ({result.error_code}): {result.error_string}"
            )

    def follow_path(
        self, points: list, cancel: CancelToken, gains: TrackingGains = None
    ):
        """Base joints in `points`: travel from the start, in the start base frame."""
        if not points:
            raise ValueError("no waypoints")
        self._ensure_mode("velocity", cancel)
        with_base = any(j in points[-1] for j in BASE_JOINTS)
        if with_base:
            points = [{**dict.fromkeys(BASE_JOINTS, 0.0), **p} for p in points]
        base_start = self.odom_pose(max_age=0.5) if with_base else None

        def measured():
            joints = self.arm_joints(max_age=0.5)
            if with_base:
                pose = self.odom_pose(max_age=0.5)
                joints[BASE_X], joints[BASE_Y], joints[BASE_THETA] = displacement(
                    base_start, pose
                )
            return joints

        tracker = PathTracker(measured(), points, gains or TrackingGains())
        arm = [j for j in tracker.joints if j not in BASE_JOINTS]
        deadline = tracker.duration + self._path_settle_timeout
        t0 = time.monotonic()
        self._arm_moving = True
        self._base_moving = with_base
        try:
            while True:
                cancel.check()
                t = time.monotonic() - t0
                current = measured()
                velocity, error, done = tracker.command(t, current)
                if done:
                    return
                if t > deadline:
                    far = {j: round(e, 4) for j, e in error.items() if abs(e) > 1e-3}
                    raise MotionError(f"path did not arrive, error {far}")
                ahead, _ = tracker.desired(t + self._period)
                self._publish_joints(
                    {
                        j: velocity[j] if j in PRISMATIC else ahead[j] - current[j]
                        for j in arm
                    }
                )
                if with_base:
                    # The path is in the start base frame, /cmd_vel in the current one.
                    turn = current[BASE_THETA]
                    vx, vy = velocity[BASE_X], velocity[BASE_Y]
                    self._publish_twist(
                        math.cos(turn) * vx + math.sin(turn) * vy,
                        -math.sin(turn) * vx + math.cos(turn) * vy,
                        velocity[BASE_THETA],
                    )
                time.sleep(self._period)
        finally:
            self._publish_joints(dict.fromkeys(ARM_JOINT_NAMES, 0.0))
            self._arm_moving = False
            if with_base:
                self._publish_twist(0.0, 0.0, 0.0)
                self._base_moving = False

    def _publish_joints(self, commands: dict):
        msg = JointJog()
        msg.joint_names = [self._driver_names[j] for j in commands]
        msg.velocities = [float(v) for v in commands.values()]
        msg.duration = 1.0
        self._joint_vel.publish(msg)

    def _ensure_mode(self, mode: str, cancel: CancelToken, timeout: float = 2.0):
        if self._mode == mode:
            return
        self._node.get_logger().info(
            f"switching stretch_driver from {self._mode} to {mode} mode"
        )
        if not self._driver_set_params.wait_for_service(timeout_sec=timeout):
            raise MotionError("/stretch_driver/set_parameters not available")
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=mode)
        future = self._driver_set_params.call_async(
            SetParameters.Request(parameters=[Parameter(name="mode", value=value)])
        )
        if not cancel.wait_until(lambda: self._mode == mode, timeout):
            result = future.result() if future.done() else None
            raise MotionError(f"stretch_driver did not switch to {mode} ({result})")

    def stop(self):
        goal = self._active_goal
        if goal is not None:
            goal.cancel_goal_async()
        if self._arm_moving:
            self._publish_joints(dict.fromkeys(ARM_JOINT_NAMES, 0.0))
        if self._base_moving:
            self._publish_twist(0.0, 0.0, 0.0)

    def move_base(
        self,
        base_x: float,
        base_y: float,
        base_theta: float,
        cancel: CancelToken,
        gains: BaseGains = None,
        rate_hz: float = 20.0,
    ):
        """Translate by (x, y), then turn by theta: the order of the IK base joints."""
        gains = gains or BaseGains()
        start = self.odom_pose(max_age=0.5)
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
                    cancel.check()
                    if time.monotonic() > deadline:
                        raise MotionError(
                            f"base motion timed out after {timeout:.0f} s"
                        )
                    vx, vy, w, done = command(self.odom_pose(max_age=0.5))
                    if done:
                        break
                    self._publish_twist(vx, vy, w)
                    time.sleep(1.0 / rate_hz)
                self._publish_twist(0.0, 0.0, 0.0)
        finally:
            self._publish_twist(0.0, 0.0, 0.0)
            self._base_moving = False

    def odom_pose(self, max_age: float) -> Pose2D:
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

    def set_gripper(self, gripper: "GripperSettings"):
        self._gripper = gripper

    # stretch_driver ends a gripper goal only within 0.5 of the target. At a
    # mechanical stop or on an object it never gets there, so wait until the
    # fingers stop.

    def open_gripper(self, cancel: CancelToken):
        self._run_gripper(self._gripper.open, cancel)

    def close_gripper(self, cancel: CancelToken) -> bool:
        """True if it holds something. The command stays on to keep the grip."""
        self._run_gripper(self._gripper.closed, cancel)
        finger = self._finger()
        holding = (
            abs(finger - self._gripper.closed_finger) > self._gripper.holding_margin
        )
        self._node.get_logger().info(
            f"gripper closed at finger {finger:.3f} (empty {self._gripper.closed_finger:.3f}): "
            f"{'holding something' if holding else 'nothing in it'}"
        )
        return holding

    def _run_gripper(self, target: float, cancel: CancelToken) -> bool:
        g = self._gripper
        goal = self._send_gripper(target, cancel)
        result = goal.get_result_async()
        start = time.monotonic()
        last, still_since, moved = self._finger(), start, False
        while True:
            if cancel.canceled:
                goal.cancel_goal_async()
                raise Canceled()
            if result.done():
                outcome = result.result().result
                if outcome.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
                    return True
                raise MotionError(
                    f"gripper to {target} failed ({outcome.error_code}): "
                    f"{outcome.error_string}"
                )
            now = time.monotonic()
            position = self._finger()
            if abs(position - last) > g.still:
                last, still_since, moved = position, now, True
            elif now - still_since >= g.stall_time and (moved or now - start > 1.0):
                return False
            if now - start > g.timeout:
                raise MotionError(f"gripper did not reach {target} in {g.timeout} s")
            time.sleep(POLL)

    def _send_gripper(self, target: float, cancel: CancelToken):
        self._ensure_mode(self._fjt_mode, cancel)
        if not self._fjt.wait_for_server(timeout_sec=2.0):
            raise MotionError("FollowJointTrajectory server not available")
        name = self._gripper.joint
        traj = joint_trajectory([name], [{name: target}], [1.0])
        goal = cancel.result(
            self._fjt.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj)),
            on_cancel=self.stop,
        )
        if not goal.accepted:
            raise MotionError("gripper goal rejected")
        return goal

    def _finger(self) -> float:
        with self._lock:
            position = self._joint_state.get(self._gripper.state_joint)
        if position is None:
            raise MotionError(f"{self._gripper.state_joint} not in /joint_states")
        return position
