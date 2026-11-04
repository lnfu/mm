"""Manipulation action servers on the robot: ~/grasp (mm_interfaces/action/Grasp).

  ros2 run mm_manip manipulation_server --ros-args --params-file config/manipulation.yaml
  ros2 action send_goal /mm_manip/grasp mm_interfaces/action/Grasp "{prompt: cup}"

dry_run: nothing moves. The plan is shown in RViz through ~/target_markers and
~/ghost_joint_states (feed it to a second robot_state_publisher with
frame_prefix "ghost/"). The ghost shows the arm only, not the base motion.
"""

import threading

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from mm_interfaces.action import Grasp

from .config import GraspConfig, ServerConfig
from .errors import Canceled
from .motion.planning import PlanningError
from .motion.robot import MotionError, Robot
from .motion.trajectory import urdf_joint_positions
from .perception.locator import Locator, PerceptionError
from .scene import Scene, SceneError
from .skills import grasp
from .skills.grasp import GraspFailed, GraspGoal, Phase
from .skills.primitives import Context, OutOfReach, sleep

assert all(getattr(Grasp.Feedback, p.name) == p for p in Phase), (
    "grasp.Phase does not match Grasp.action"
)


class ManipulationServer(Node):
    def __init__(self):
        super().__init__("mm_manip")
        ServerConfig.declare(self)
        GraspConfig.declare(self)
        cfg = ServerConfig.from_node(self)

        group = ReentrantCallbackGroup()
        self._robot = Robot(
            self,
            cfg.fjt_action,
            cfg.driver_joint,
            max_step=cfg.max_step,
            no_split_action_timeout=cfg.no_split_action_timeout,
            callback_group=group,
        )
        self._scene = Scene(self, base_frame=cfg.base_frame)
        self._locator = Locator(
            self,
            self._scene,
            cfg.detect_service,
            cfg.color_topic,
            cfg.depth_topic,
            cfg.camera_info_topic,
            depth_compressed=cfg.depth_compressed,
            sync_slop=cfg.sync_slop,
            snapshot_timeout=cfg.snapshot_timeout,
            service_timeout=cfg.service_timeout,
            depth_band=cfg.depth_band,
            callback_group=group,
        )
        self._ghost_pub = self.create_publisher(JointState, "~/ghost_joint_states", 1)

        self._busy = False
        self._busy_lock = threading.Lock()
        ActionServer(
            self,
            Grasp,
            "~/grasp",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.get_logger().info(f"READY (dry_run={cfg.dry_run})")

    def _on_goal(self, goal):
        if goal.mode == Grasp.Goal.MODE_PROMPT and not goal.prompt.strip():
            self.get_logger().warn("rejecting goal: MODE_PROMPT needs a prompt")
            return GoalResponse.REJECT
        if goal.mode not in (Grasp.Goal.MODE_PROMPT, Grasp.Goal.MODE_POINT):
            self.get_logger().warn(f"rejecting goal: unknown mode {goal.mode}")
            return GoalResponse.REJECT
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn("rejecting goal: already grasping")
                return GoalResponse.REJECT
            self._busy = True
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        result = Grasp.Result()
        cfg = GraspConfig.from_node(self)
        shared = ServerConfig.from_node(self)

        def should_cancel():
            return goal_handle.is_cancel_requested

        ctx = Context(
            robot=self._robot,
            locator=self._locator,
            observer=_Observer(self, goal_handle, shared.base_frame, should_cancel),
            log=self.get_logger(),
            should_cancel=should_cancel,
        )
        try:
            center = grasp.run(self._goal(goal_handle.request), cfg, shared, ctx)
            result.success = True
            if shared.dry_run:
                result.message = "dry_run: plan ok"
            elif shared.use_gripper:
                result.message = "grasped"
            else:
                result.message = "motion done (use_gripper is false, gripper not used)"
            result.grasped_point = self._point_stamped(shared.base_frame, center)
            goal_handle.succeed()
        except Canceled:
            self._robot.stop()
            result.message = "canceled"
            goal_handle.canceled()
        except NotImplementedError as e:
            result.message = f"not implemented yet: {e}"
            goal_handle.abort()
        except (
            GraspFailed,
            OutOfReach,
            MotionError,
            PlanningError,
            SceneError,
            PerceptionError,
        ) as e:
            result.message = str(e)
            goal_handle.abort()
        finally:
            with self._busy_lock:
                self._busy = False

        if result.success:
            self.get_logger().info(f"grasp result: {result.message}")
        else:
            self.get_logger().warn(f"grasp result: {result.message}")
        return result

    def _goal(self, request):
        """The action goal as a GraspGoal; a MODE_POINT target goes to the base frame."""
        if request.mode == Grasp.Goal.MODE_POINT:
            point = request.target_point
            return GraspGoal(point=self._scene.to_base(point.header, point.point))
        return GraspGoal(prompt=request.prompt)

    def _point_stamped(self, frame, center):
        msg = PointStamped()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = (float(v) for v in center)
        return msg


class _Observer:
    """What grasp.run reports, as action feedback, RViz markers and logs."""

    def __init__(self, server, goal_handle, base_frame, should_cancel):
        self._server = server
        self._goal_handle = goal_handle
        self._base_frame = base_frame
        self._should_cancel = should_cancel

    def phase(self, phase):
        self._goal_handle.publish_feedback(Grasp.Feedback(phase=int(phase)))

    def target(self, center):
        # Shown before planning, so it is visible even if IK fails.
        self._server._scene.publish_markers(center)
        self._log(f"target in {self._base_frame}: {center.round(3)}")

    def plan(self, center, plan):
        self._server._scene.publish_markers(
            center, plan.poses["approach"] + [plan.poses["lift"]]
        )
        self._log(f"plan ok: base {plan.base or 'locked'}, center {center.round(3)}")

    def ghost(self, waypoints):
        """Play the arm waypoints on ~/ghost_joint_states, using URDF joint names."""
        for joints in waypoints:
            msg = JointState()
            msg.header.stamp = self._server.get_clock().now().to_msg()
            msg.name, msg.position = urdf_joint_positions(joints)
            self._server._ghost_pub.publish(msg)
            sleep(0.3, self._should_cancel)

    def _log(self, message):
        self._server.get_logger().info(message)


def main():
    rclpy.init()
    node = ManipulationServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
