"""Manipulation action servers on the robot, one goal at a time across both:
~/grasp (mm_interfaces/action/Grasp) and ~/articulate (mm_interfaces/action/Articulate).

  ros2 run mm_manip manipulation_server --ros-args --params-file config/manipulation.yaml
  ros2 action send_goal /mm_manip/grasp mm_interfaces/action/Grasp "{prompt: cup}"
  ros2 action send_goal /mm_manip/articulate mm_interfaces/action/Articulate \
    "{prompt: the cabinet door}"

MODE_PROMPT of articulate asks Gemini, with the key in GEMINI_API_KEY.

dry_run: nothing moves. The plan is shown in RViz through ~/target_markers and
~/ghost_joint_states (feed it to a second robot_state_publisher with
frame_prefix "ghost/"). The ghost shows the arm only, not the base motion.
"""

import threading
import traceback

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from mm_interfaces.action import Articulate, Grasp

from .config import ArticulateConfig, GraspConfig, ServerConfig
from .errors import Canceled
from .motion.planning import PlanningError
from .motion.robot import MotionError, Robot
from .motion.trajectory import urdf_joint_positions
from .perception.door import DoorType, GeminiDoorKeypoints, HandleType
from .perception.door_locator import DoorLocator
from .perception.gemini import GeminiClient, GeminiError
from .perception.locator import Locator, PerceptionError
from .scene import Scene, SceneError
from .skills import articulate, grasp
from .skills.articulate import ArticulateFailed, ArticulateGoal
from .skills.grasp import GraspFailed, GraspGoal
from .skills.primitives import Context, OutOfReach, sleep

for _phases, _feedback in (
    (grasp.Phase, Grasp.Feedback),
    (articulate.Phase, Articulate.Feedback),
):
    assert all(getattr(_feedback, p.name) == p for p in _phases), (
        f"{_phases.__module__} Phase does not match the action"
    )
assert all(getattr(Articulate.Result, f"TYPE_{t.name}") == t for t in DoorType)
assert all(getattr(Articulate.Result, f"HANDLE_{h.name}") == h for h in HandleType)

# Errors that abort a goal with their message.
_FAILURES = (
    GraspFailed,
    ArticulateFailed,
    OutOfReach,
    MotionError,
    PlanningError,
    SceneError,
    PerceptionError,
)


class ManipulationServer(Node):
    def __init__(self):
        super().__init__("mm_manip")
        ServerConfig.declare(self)
        GraspConfig.declare(self)
        ArticulateConfig.declare(self)
        cfg = ServerConfig.from_node(self)

        group = ReentrantCallbackGroup()
        self._robot = Robot(
            self,
            cfg.fjt_action,
            cfg.driver_joint,
            max_step=cfg.max_step,
            no_split_action_timeout=cfg.no_split_action_timeout,
            fjt_mode=cfg.fjt_mode,
            control_rate=cfg.control_rate,
            path_settle_timeout=cfg.path_settle_timeout,
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
        self._door_locator = DoorLocator(
            self,
            self._scene,
            cfg.color_topic,
            cfg.depth_topic,
            cfg.camera_info_topic,
            depth_compressed=cfg.depth_compressed,
            sync_slop=cfg.sync_slop,
            snapshot_timeout=cfg.snapshot_timeout,
            callback_group=group,
        )
        self._ghost_pub = self.create_publisher(JointState, "~/ghost_joint_states", 1)

        self._busy = False
        self._busy_lock = threading.Lock()
        for action, name, on_goal, execute in (
            (Grasp, "~/grasp", self._on_grasp_goal, self._execute_grasp),
            (
                Articulate,
                "~/articulate",
                self._on_articulate_goal,
                self._execute_articulate,
            ),
        ):
            ActionServer(
                self,
                action,
                name,
                execute_callback=execute,
                goal_callback=on_goal,
                cancel_callback=lambda _: CancelResponse.ACCEPT,
                callback_group=group,
            )
        self.get_logger().info(f"READY (dry_run={cfg.dry_run})")

    # --- goals --------------------------------------------------------------

    def _accept(self, goal, action, what) -> GoalResponse:
        """Accept a goal of `action` if it is well formed and nothing else runs."""
        if goal.mode == action.Goal.MODE_PROMPT and not goal.prompt.strip():
            self.get_logger().warn(f"rejecting {what} goal: MODE_PROMPT needs a prompt")
            return GoalResponse.REJECT
        if goal.mode not in (action.Goal.MODE_PROMPT, action.Goal.MODE_POINT):
            self.get_logger().warn(f"rejecting {what} goal: unknown mode {goal.mode}")
            return GoalResponse.REJECT
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(f"rejecting {what} goal: a goal is running")
                return GoalResponse.REJECT
            self._busy = True
        return GoalResponse.ACCEPT

    def _on_grasp_goal(self, goal):
        return self._accept(goal, Grasp, "grasp")

    def _on_articulate_goal(self, goal):
        return self._accept(goal, Articulate, "articulate")

    def _skill_context(self, goal_handle, observer_type, shared, locator):
        # Not "_context": rclpy.Node keeps its rclpy Context in self._context.
        def should_cancel():
            return goal_handle.is_cancel_requested

        return Context(
            robot=self._robot,
            locator=locator,
            observer=observer_type(self, goal_handle, shared.base_frame, should_cancel),
            log=self.get_logger(),
            should_cancel=should_cancel,
        )

    def _finish(self, goal_handle, result, what, body):
        """Run body() (sets result.message on success), end the goal, free the robot."""
        try:
            body()
            result.success = True
            goal_handle.succeed()
        except Canceled:
            self._robot.stop()
            result.message = "canceled"
            goal_handle.canceled()
        except NotImplementedError as e:
            result.message = f"not implemented yet: {e}"
            goal_handle.abort()
        except _FAILURES as e:
            result.message = str(e)
            goal_handle.abort()
        except Exception as e:  # a bug: stop, report it, and stay usable
            self._robot.stop()
            self.get_logger().error(f"{what} crashed: {traceback.format_exc()}")
            result.message = f"internal error: {e!r}"
            goal_handle.abort()
        finally:
            with self._busy_lock:
                self._busy = False

        log = self.get_logger()
        (log.info if result.success else log.warn)(f"{what} result: {result.message}")
        return result

    def _execute_grasp(self, goal_handle):
        result = Grasp.Result()

        def body():
            # Everything inside body(), so _finish frees the robot whatever fails.
            cfg = GraspConfig.from_node(self)
            shared = ServerConfig.from_node(self)
            ctx = self._skill_context(
                goal_handle, _GraspObserver, shared, self._locator
            )
            center = grasp.run(self._grasp_goal(goal_handle.request), cfg, shared, ctx)
            result.message = _motion_message(shared, "grasped")
            result.grasped_point = self._point_stamped(shared.base_frame, center)

        return self._finish(goal_handle, result, "grasp", body)

    def _execute_articulate(self, goal_handle):
        result = Articulate.Result()

        def body():
            cfg = ArticulateConfig.from_node(self)
            shared = ServerConfig.from_node(self)
            locator = _DoorFinder(self._door_locator, cfg)
            ctx = self._skill_context(goal_handle, _ArticulateObserver, shared, locator)
            ctx.observer.result = result  # filled with the door and handle type
            goal = self._articulate_goal(goal_handle.request)
            outcome = articulate.run(goal, cfg, shared, ctx)
            opened, asked = np.rad2deg(outcome.opened), np.rad2deg(outcome.asked)
            result.message = (
                f"{_motion_message(shared, 'opened')}: {opened:.0f} of {asked:.0f} deg"
            )

        return self._finish(goal_handle, result, "articulate", body)

    def _articulate_goal(self, request):
        """The action goal as an ArticulateGoal; MODE_POINT points go to the base frame."""
        if request.mode == Articulate.Goal.MODE_POINT:
            to_base = self._scene.to_base
            return ArticulateGoal(
                handle=to_base(request.handle.header, request.handle.point),
                hinge=to_base(request.hinge.header, request.hinge.point),
                open_angle=request.open_angle,
            )
        return ArticulateGoal(prompt=request.prompt, open_angle=request.open_angle)

    def _grasp_goal(self, request):
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


def _motion_message(shared, done: str) -> str:
    if shared.dry_run:
        return "dry_run: plan ok"
    if shared.use_gripper:
        return done
    return "motion done (use_gripper is false, gripper not used)"


class _DoorFinder:
    """ctx.locator for articulate: the door Gemini points to, for this goal's config."""

    def __init__(self, door_locator, cfg):
        self._door_locator = door_locator
        self._cfg = cfg

    def locate(self, prompt, should_cancel):
        cfg = self._cfg
        try:
            client = GeminiClient(
                cfg.gemini_model, cfg.gemini_thinking_level, cfg.gemini_timeout
            )
        except GeminiError as e:
            raise PerceptionError(f"gemini: {e}") from e
        return self._door_locator.locate(
            prompt,
            GeminiDoorKeypoints(client, max_side=cfg.image_max_side),
            should_cancel,
            depth_band=cfg.door_depth_band,
            shrink=cfg.quad_shrink,
        )


class _Observer:
    """What a skill reports, as action feedback, RViz markers and logs."""

    feedback = None  # the action's Feedback type

    def __init__(self, server, goal_handle, base_frame, should_cancel):
        self._server = server
        self._goal_handle = goal_handle
        self._base_frame = base_frame
        self._should_cancel = should_cancel

    def phase(self, phase):
        self._goal_handle.publish_feedback(self.feedback(phase=int(phase)))

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


class _ArticulateObserver(_Observer):
    feedback = Articulate.Feedback
    result = None  # Articulate.Result to fill with what was detected

    def door(self, door):
        if self.result is not None:
            self.result.type = int(door.type)
            self.result.handle_type = int(door.handle_type)
        self._server._scene.publish_markers(door.handle)
        self._log(
            f"door {door.type.name}, handle {door.handle_type.name} "
            f"at {door.handle.round(3)} in {self._base_frame}"
        )

    def arc(self, arc):
        self._server._scene.publish_markers(arc.axis, arc.poses)
        self._log(f"hinge axis at {arc.axis.round(3)}, {len(arc.poses)} arc poses")


class _GraspObserver(_Observer):
    feedback = Grasp.Feedback

    def target(self, center):
        # Shown before planning, so it is visible even if IK fails.
        self._server._scene.publish_markers(center)
        self._log(f"target in {self._base_frame}: {center.round(3)}")

    def plan(self, center, plan):
        self._server._scene.publish_markers(
            center, plan.poses["approach"] + [plan.poses["lift"]]
        )
        self._log(f"plan ok: base {plan.base or 'locked'}, center {center.round(3)}")


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
