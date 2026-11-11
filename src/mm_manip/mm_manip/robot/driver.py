from contextlib import contextmanager

from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointJog
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient

from mm_ik.joints import ARM_JOINT_NAMES

from ..cancel import CancelToken
from ..errors import MotionError
from ..motion.trajectory import joint_trajectory
from .state import RobotState


class Driver:
    """stretch_driver: its mode, FollowJointTrajectory, /joint_vel and /cmd_vel."""

    def __init__(
        self,
        node,
        state: RobotState,
        fjt_action: str,
        driver_joint_names: dict,
        callback_group=None,
    ):
        self._node = node
        self._state = state
        self._names = driver_joint_names
        self._fjt = ActionClient(
            node, FollowJointTrajectory, fjt_action, callback_group=callback_group
        )
        self._get_params = node.create_client(
            GetParameters,
            "/stretch_driver/get_parameters",
            callback_group=callback_group,
        )
        self._set_params = node.create_client(
            SetParameters,
            "/stretch_driver/set_parameters",
            callback_group=callback_group,
        )
        self._joint_vel = node.create_publisher(JointJog, "/joint_vel", 1)
        # stretch_driver has no /cmd_vel timeout. Always send zero to stop the base.
        self._cmd_vel = node.create_publisher(Twist, "/cmd_vel", 1)
        self._active_goal = None
        self._arm_streaming = False
        self._base_streaming = False

    def driver_name(self, joint: str) -> str:
        return self._names[joint]

    def ensure_mode(self, mode: str, cancel: CancelToken, timeout: float = 2.0):
        if self._state.mode == mode:
            return
        self._node.get_logger().info(
            f"switching stretch_driver from {self._state.mode} to {mode} mode"
        )
        if not self._set_params.wait_for_service(timeout_sec=timeout):
            raise MotionError("/stretch_driver/set_parameters not available")
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=mode)
        future = self._set_params.call_async(
            SetParameters.Request(parameters=[Parameter(name="mode", value=value)])
        )
        if not cancel.wait_until(lambda: self._state.mode == mode, timeout):
            result = future.result() if future.done() else None
            raise MotionError(f"stretch_driver did not switch to {mode} ({result})")

    def action_timeout(self, cancel: CancelToken, timeout: float = 2.0) -> float | None:
        """stretch_driver's action_timeout (s), or None if it does not answer."""
        if not self._get_params.wait_for_service(timeout_sec=timeout):
            return None
        future = self._get_params.call_async(
            GetParameters.Request(names=["action_timeout"])
        )
        if cancel.wait_until(future.done, timeout) and future.result().values:
            return future.result().values[0].double_value
        return None

    def send_point(self, positions: dict, cancel: CancelToken, what="trajectory"):
        """Send one FollowJointTrajectory point (driver names). Returns the goal."""
        if not self._fjt.wait_for_server(timeout_sec=2.0):
            raise MotionError("FollowJointTrajectory server not available")
        names = list(positions)
        traj = joint_trajectory(names, [positions], [1.0])
        goal = cancel.result(
            self._fjt.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj)),
            on_cancel=self.stop,
        )
        if not goal.accepted:
            raise MotionError(f"{what} rejected")
        return goal

    def wait_point(self, goal, cancel: CancelToken):
        """Wait for a goal from send_point; stop() cancels it meanwhile."""
        self._active_goal = goal
        try:
            result = cancel.result(goal.get_result_async(), on_cancel=self.stop).result
        finally:
            self._active_goal = None
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise MotionError(
                f"trajectory failed ({result.error_code}): {result.error_string}"
            )

    @contextmanager
    def streaming(self, arm: bool, base: bool):
        """Velocity commands inside; zero velocity on exit and on stop()."""
        self._arm_streaming, self._base_streaming = arm, base
        try:
            yield
        finally:
            if arm:
                self.joint_velocities(dict.fromkeys(ARM_JOINT_NAMES, 0.0))
            if base:
                self.twist(0.0, 0.0, 0.0)
            self._arm_streaming = self._base_streaming = False

    def joint_velocities(self, commands: dict):
        msg = JointJog()
        msg.joint_names = [self._names[j] for j in commands]
        msg.velocities = [float(v) for v in commands.values()]
        msg.duration = 1.0
        self._joint_vel.publish(msg)

    def twist(self, vx, vy, w):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(w)
        self._cmd_vel.publish(msg)

    def stop(self):
        goal = self._active_goal
        if goal is not None:
            goal.cancel_goal_async()
        if self._arm_streaming:
            self.joint_velocities(dict.fromkeys(ARM_JOINT_NAMES, 0.0))
        if self._base_streaming:
            self.twist(0.0, 0.0, 0.0)
