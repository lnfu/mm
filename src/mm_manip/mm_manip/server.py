"""Action server: one ROS action per skill, one goal at a time. See NOTE.md."""

import threading
import traceback
from functools import partial

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .cancel import CancelToken
from .config import ServerConfig
from .errors import Canceled, SkillError
from .runtime import build, gripper_settings
from .skills.action import SkillAction
from .skills.articulate.action import ArticulateAction
from .skills.base import Context
from .skills.grasp.action import GraspAction

SKILLS = (GraspAction, ArticulateAction)


class ManipulationServer(Node):
    def __init__(self, skills=SKILLS):
        super().__init__("mm_manip")
        ServerConfig.declare(self)
        for skill in skills:
            skill.config.declare(self)
        shared = ServerConfig.from_node(self)
        self._runtime = build(self, shared)
        self._busy = False
        self._busy_lock = threading.Lock()
        for skill_type in skills:
            skill = skill_type(self, self._runtime, shared)
            _check_phases(skill)
            ActionServer(
                self,
                skill.action,
                f"~/{skill.name}",
                execute_callback=partial(self._execute, skill),
                goal_callback=partial(self._on_goal, skill),
                cancel_callback=lambda _: CancelResponse.ACCEPT,
                callback_group=self._runtime.callback_group,
            )
        self.get_logger().info(f"READY (dry_run={shared.dry_run})")

    def _on_goal(self, skill: SkillAction, goal) -> GoalResponse:
        problem = skill.reject(goal)
        if problem is None:
            with self._busy_lock:
                if self._busy:
                    problem = "a goal is running"
                else:
                    self._busy = True
        if problem is None:
            return GoalResponse.ACCEPT
        self.get_logger().warn(f"rejecting {skill.name} goal: {problem}")
        return GoalResponse.REJECT

    def _execute(self, skill: SkillAction, goal_handle):
        robot = self._runtime.robot
        result = skill.action.Result()
        cancel = CancelToken(lambda: goal_handle.is_cancel_requested)
        report = _GoalReport(goal_handle, skill.action.Feedback, self._runtime, cancel)
        outcome = None
        try:
            shared = ServerConfig.from_node(self)
            cfg = skill.config.from_node(self)
            robot.set_gripper(gripper_settings(shared))
            ctx = Context(robot, skill.locator(cfg), report, self.get_logger(), cancel)
            outcome = skill.run(skill.goal(goal_handle.request), cfg, shared, ctx)
            result.message = skill.message(outcome, shared)
            result.success = True
            goal_handle.succeed()
        except Canceled:
            robot.stop()
            result.message = "canceled"
            goal_handle.canceled()
        except NotImplementedError as e:
            result.message = f"not implemented yet: {e}"
            goal_handle.abort()
        except SkillError as e:
            result.message = str(e)
            goal_handle.abort()
        except Exception as e:  # a bug: stop, log, and stay usable
            robot.stop()
            self.get_logger().error(f"{skill.name} crashed: {traceback.format_exc()}")
            result.message = f"internal error: {e!r}"
            goal_handle.abort()
        finally:
            with self._busy_lock:
                self._busy = False
        skill.fill_result(result, outcome, report.kept)

        # rclpy refuses a call site that logs with two severities.
        if result.success:
            self.get_logger().info(f"{skill.name} result: {result.message}")
        else:
            self.get_logger().warn(f"{skill.name} result: {result.message}")
        return result


def _check_phases(skill: SkillAction):
    feedback = skill.action.Feedback
    for phase in skill.phases:
        if getattr(feedback, phase.name, None) != phase:
            raise RuntimeError(f"{skill.name} phase {phase.name} does not match")


class _GoalReport:
    def __init__(self, goal_handle, feedback_type, runtime, cancel: CancelToken):
        self._goal_handle = goal_handle
        self._feedback_type = feedback_type
        self._scene = runtime.scene
        self._cancel = cancel
        self.kept = {}

    def phase(self, phase):
        self._goal_handle.publish_feedback(self._feedback_type(phase=int(phase)))

    def show(self, center, poses=()):
        self._scene.publish_markers(center, poses)

    def ghost(self, waypoints):
        for joints in waypoints:
            self._scene.publish_ghost(joints)
            self._cancel.sleep(0.3)

    def keep(self, **values):
        self.kept.update(values)


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
