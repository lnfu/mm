"""Hardware interface to stretch_driver."""

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

from ..errors import Canceled
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

# Prismatic joints take a velocity on /joint_vel. stretch_driver turns a wrist
# "velocity" v into move_by(v * duration) from the measured position, so for the
# wrist follow_path sends, with duration 1 s, the remaining distance to the path.
PRISMATIC = ("lift_joint", "arm_joint")


class MotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GripperSettings:
    joint: str = "gripper_joint"  # FollowJointTrajectory command (servo rad)
    state_joint: str = "gripper_finger_left_joint"  # /joint_states (finger rad)
    # Driver units. Both are past the mechanical stops, so the gripper never gets
    # within the driver's 0.5 of them and moves end when the fingers stop.
    open: float = 80.0
    closed: float = -10.0
    # The finger reading (finger rad) when closed on nothing; closing ends farther
    # than holding_margin from it when something is between the fingers.
    closed_finger: float = 0.0
    holding_margin: float = 0.05
    still: float = 0.005  # finger rad: smaller changes count as not moving
    stall_time: float = 0.3  # s still, short of closed, to count as holding something
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
        """
        Args:
            fjt_action: FollowJointTrajectory action name.
            driver_joint_names: IK joint name (ARM_JOINT_NAMES) -> name the driver uses
                in joint_states and FollowJointTrajectory.
            max_step: largest change per joint in one trajectory point, used when the
                driver's action_timeout is shorter than no_split_action_timeout.
            fjt_mode: driver mode for move_arm ("navigation" or "position"; both
                take FollowJointTrajectory, only navigation also takes /cmd_vel).
            control_rate: follow_path command rate (Hz).
            path_settle_timeout: time follow_path allows after the path ends to get
                within tolerance.
            gripper: its joint names, open and closed commands, stall detection.
        """
        self._node = node
        self._driver_names = driver_joint_names
        self._max_step = max_step
        self._no_split_action_timeout = no_split_action_timeout
        self._split = None  # last answer of _needs_split, to log only changes
        self._fjt_mode = fjt_mode
        self._period = 1.0 / control_rate
        self._path_settle_timeout = path_settle_timeout
        self._gripper = gripper or GripperSettings()
        self._lock = threading.Lock()
        self._joint_state = {}
        self._joint_state_time = None
        self._mode = None  # latest /mode from the driver
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
        self._arm_moving = False  # follow_path is commanding /joint_vel

        # Base: stretch_driver takes /cmd_vel (vx, vy, w) in navigation and velocity
        # modes and has no cmd_vel timeout, so every base motion ends by publishing zero.
        self._odom = None  # (Pose2D, receive time)
        node.create_subscription(
            Odometry, "/wheel_odom", self._on_odom, 10, callback_group=callback_group
        )
        self._cmd_vel = node.create_publisher(Twist, "/cmd_vel", 1)
        self._base_moving = False

    # --- state ------------------------------------------------------------

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
        """Current arm joints keyed by IK names.

        The arm is read from its driver name if published, otherwise summed from
        the URDF segments arm_l1..arm_l4 (stretch_driver publishes the segments,
        each at a quarter of the total). With max_age, older joint states raise.
        """
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
        self._ensure_mode(self._fjt_mode, should_cancel)
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

    def follow_path(self, points: list, should_cancel, gains: TrackingGains = None):
        """Move through `points` without stopping at them, and wait.

        points: dicts keyed by IK joint names, and optionally BASE_X, BASE_Y and
        BASE_THETA: the base's travel (m) from where it is when the path starts,
        along its x and y axes there, and its turn (rad) since then. A path with
        base joints controls all three (the missing ones are held at 0). gains: path timing, tracking gain and arrival tolerances
        (per call, so a skill can read them from its parameters on every goal).
        The path starts from the measured joints and ends once
        every joint is within tolerance of the last point; everything is always
        commanded to zero at the end.
        Raises Canceled if should_cancel() turns true, MotionError if the driver
        does not switch to velocity mode, joint states or odometry stop, or the
        robot does not arrive.
        """
        if not points:
            raise ValueError("no waypoints")
        self._ensure_mode("velocity", should_cancel)
        with_base = any(j in points[-1] for j in BASE_JOINTS)
        if with_base:
            points = [{**dict.fromkeys(BASE_JOINTS, 0.0), **p} for p in points]
        base_start = self._odom_pose(max_age=0.5) if with_base else None

        def measured():
            joints = self.arm_joints(max_age=0.5)
            if with_base:
                pose = self._odom_pose(max_age=0.5)
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
                if should_cancel():
                    raise Canceled()
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
                    # The path's (x, y) is in the starting base frame, /cmd_vel in
                    # the current one, turned by base_theta since the start.
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
        msg.duration = 1.0  # wrist: move_by(velocity * duration), see PRISMATIC
        self._joint_vel.publish(msg)

    def _ensure_mode(self, mode: str, should_cancel, timeout: float = 2.0):
        """Switch stretch_driver to `mode` (its `mode` parameter) unless it is in it."""
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
        deadline = time.monotonic() + timeout
        while self._mode != mode:
            if should_cancel():
                raise Canceled()
            if time.monotonic() > deadline:
                result = future.result() if future.done() else None
                raise MotionError(f"stretch_driver did not switch to {mode} ({result})")
            time.sleep(0.02)

    def stop(self):
        """Stop all motion now."""
        goal = self._active_goal
        if goal is not None:
            goal.cancel_goal_async()
        if self._arm_moving:
            self._publish_joints(dict.fromkeys(ARM_JOINT_NAMES, 0.0))
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
    # Through FollowJointTrajectory. The driver counts the gripper as there only
    # within 0.5 of its target, which it never is against its mechanical stops or
    # on an object: such a goal only ends at the driver's action_timeout. So both
    # wait for the fingers to stop moving instead.

    def open_gripper(self, should_cancel):
        """Open the gripper; returns once it is open (at the target or the stop)."""
        self._run_gripper(self._gripper.open, should_cancel)

    def close_gripper(self, should_cancel) -> bool:
        """Close the gripper; True if it stopped on something, False if it closed
        all the way (nothing in it). The command is left running, so the gripper
        keeps squeezing."""
        self._run_gripper(self._gripper.closed, should_cancel)
        finger = self._finger()
        holding = (
            abs(finger - self._gripper.closed_finger) > self._gripper.holding_margin
        )
        self._node.get_logger().info(
            f"gripper closed at finger {finger:.3f} (empty {self._gripper.closed_finger:.3f}): "
            f"{'holding something' if holding else 'nothing in it'}"
        )
        return holding

    def _run_gripper(self, target: float, should_cancel) -> bool:
        """Command the gripper to `target`; True once the driver reports it there,
        False once the fingers stop short of it (still for stall_time)."""
        g = self._gripper
        goal = self._send_gripper(target, should_cancel)
        result = goal.get_result_async()
        start = time.monotonic()
        last, still_since, moved = self._finger(), start, False
        while True:
            if should_cancel():
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
                return False  # stopped short (or never had to move)
            if now - start > g.timeout:
                raise MotionError(f"gripper did not reach {target} in {g.timeout} s")
            time.sleep(0.02)

    def _send_gripper(self, target: float, should_cancel):
        self._ensure_mode(self._fjt_mode, should_cancel)
        if not self._fjt.wait_for_server(timeout_sec=2.0):
            raise MotionError("FollowJointTrajectory server not available")
        name = self._gripper.joint
        traj = joint_trajectory([name], [{name: target}], [1.0])
        goal = self._wait(
            self._fjt.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj)),
            should_cancel,
        )
        if not goal.accepted:
            raise MotionError("gripper goal rejected")
        return goal

    def _finger(self) -> float:
        """The gripper's finger position (finger rad) from /joint_states."""
        with self._lock:
            position = self._joint_state.get(self._gripper.state_joint)
        if position is None:
            raise MotionError(f"{self._gripper.state_joint} not in /joint_states")
        return position
