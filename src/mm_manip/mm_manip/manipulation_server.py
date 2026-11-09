import threading
import traceback

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from mm_interfaces.action import Articulate, Grasp

from .cancel import CancelToken
from .config import ArticulateConfig, GraspConfig, ServerConfig
from .errors import Canceled, PerceptionError, SkillError
from .runtime import build, gripper_settings
from .skills.articulate import skill as articulate
from .skills.articulate.door import DoorType, HandleType
from .skills.articulate.gemini import GeminiClient, GeminiError
from .skills.articulate.keypoints import GeminiDoorKeypoints
from .skills.articulate.locator import DoorLocator
from .skills.articulate.skill import ArticulateGoal
from .skills.grasp import skill as grasp
from .skills.grasp.locator import Locator
from .skills.grasp.skill import GraspGoal
from .skills.primitives import Context

for _phases, _feedback in (
    (grasp.Phase, Grasp.Feedback),
    (articulate.Phase, Articulate.Feedback),
):
    assert all(getattr(_feedback, p.name) == p for p in _phases), (
        f"{_phases.__module__} Phase does not match the action"
    )
assert all(getattr(Articulate.Result, f"TYPE_{t.name}") == t for t in DoorType)
assert all(getattr(Articulate.Result, f"HANDLE_{h.name}") == h for h in HandleType)


class ManipulationServer(Node):
    def __init__(self):
        super().__init__("mm_manip")
        ServerConfig.declare(self)
        GraspConfig.declare(self)
        ArticulateConfig.declare(self)
        cfg = ServerConfig.from_node(self)

        runtime = build(self, cfg)
        group = runtime.callback_group
        self._robot, self._scene = runtime.robot, runtime.scene
        self._locator = Locator(
            self,
            runtime.camera,
            cfg.detect_service,
            service_timeout=cfg.service_timeout,
            depth_band=cfg.depth_band,
            callback_group=group,
        )
        self._door_locator = DoorLocator(self, runtime.camera, ee_frame=cfg.ee_frame)

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

    def _accept(self, goal, action, what) -> GoalResponse:
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

    # Do not rename to _context: rclpy.Node uses self._context.
    def _skill_context(self, goal_handle, observer_type, shared, locator):
        cancel = CancelToken(lambda: goal_handle.is_cancel_requested)
        self._robot.set_gripper(gripper_settings(shared))
        return Context(
            robot=self._robot,
            locator=locator,
            observer=observer_type(self, goal_handle, shared.base_frame, cancel),
            log=self.get_logger(),
            cancel=cancel,
        )

    def _finish(self, goal_handle, result, what, body):
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
        except SkillError as e:
            result.message = str(e)
            goal_handle.abort()
        except Exception as e:  # a bug: stop, log, and stay usable
            self._robot.stop()
            self.get_logger().error(f"{what} crashed: {traceback.format_exc()}")
            result.message = f"internal error: {e!r}"
            goal_handle.abort()
        finally:
            with self._busy_lock:
                self._busy = False

        # rclpy refuses a call site that logs with two severities.
        if result.success:
            self.get_logger().info(f"{what} result: {result.message}")
        else:
            self.get_logger().warn(f"{what} result: {result.message}")
        return result

    def _execute_grasp(self, goal_handle):
        result = Grasp.Result()

        # Do all work in body(), so that _finish frees the server on all errors.
        def body():
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
            ctx.observer.result = result
            goal = self._articulate_goal(goal_handle.request)
            outcome = articulate.run(goal, cfg, shared, ctx)
            opened, asked = np.rad2deg(outcome.opened), np.rad2deg(outcome.asked)
            result.message = (
                f"{_motion_message(shared, 'opened')}: {opened:.0f} of {asked:.0f} deg"
            )

        return self._finish(goal_handle, result, "articulate", body)

    def _articulate_goal(self, request):
        if request.mode == Articulate.Goal.MODE_POINT:
            to_base = self._scene.to_base
            return ArticulateGoal(
                handle=to_base(request.handle.header, request.handle.point),
                hinge=to_base(request.hinge.header, request.hinge.point),
                open_angle=request.open_angle,
            )
        return ArticulateGoal(prompt=request.prompt, open_angle=request.open_angle)

    def _grasp_goal(self, request):
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
    def __init__(self, door_locator, cfg):
        self._door_locator = door_locator
        self._cfg = cfg

    def locate(self, prompt, cancel):
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
            cancel,
            depth_band=cfg.door_depth_band,
            shrink=cfg.quad_shrink,
            debug_dir=cfg.debug_dir,
        )


class _Observer:
    feedback = None

    def __init__(self, server, goal_handle, base_frame, cancel):
        self._server = server
        self._goal_handle = goal_handle
        self._base_frame = base_frame
        self._cancel = cancel

    def phase(self, phase):
        self._goal_handle.publish_feedback(self.feedback(phase=int(phase)))

    def ghost(self, waypoints):
        for joints in waypoints:
            self._server._scene.publish_ghost(joints)
            self._cancel.sleep(0.3)

    def _log(self, message):
        self._server.get_logger().info(message)


class _ArticulateObserver(_Observer):
    feedback = Articulate.Feedback
    result = None

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
