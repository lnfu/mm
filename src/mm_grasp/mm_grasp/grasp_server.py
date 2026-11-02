"""Grasp action server (mm_interfaces/action/Grasp) on the robot.

  ros2 run mm_grasp grasp_server --ros-args --params-file config/grasp.yaml
  ros2 action send_goal /mm_grasp/grasp mm_interfaces/action/Grasp "{prompt: cup}"

Flow: start pose -> snapshot + detect + 3D -> pre-grasp -> straight approach -> close -> lift.
Success leaves the arm lifted; failure stops where it is; no retries.

dry_run: nothing moves. The plan is shown in RViz through ~/target_markers and
~/ghost_joint_states (feed it to a second robot_state_publisher with
frame_prefix "ghost/"). The ghost shows the arm only, not the base motion.
"""

import threading
import time

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import PointStamped
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from mm_interfaces.action import Grasp
from mm_interfaces.srv import Detect

from .base_control import BaseGains
from .config import GraspConfig
from .localize import center_and_width, mask_points, transform_points
from .msg_utils import decode_mask_png, depth_to_array, urdf_joint_positions
from .plan import (
    PlanningError,
    fixed_grasp_rotation,
    make_solvers,
    object_width,
    plan_grasp,
    split_long_moves,
)
from .robot import Canceled, MotionError, Robot
from .scene import Scene, SceneError
from .snapshot import SnapshotError, take_snapshot


class GraspFailed(RuntimeError):
    pass


class GraspServer(Node):
    def __init__(self):
        super().__init__("mm_grasp")
        GraspConfig.declare(self)
        cfg = GraspConfig.from_node(self)

        group = ReentrantCallbackGroup()
        self._robot = Robot(
            self, cfg.fjt_action, cfg.driver_joint, callback_group=group
        )
        self._scene = Scene(self, base_frame=cfg.base_frame)
        self._group = group
        self._driver_params = self.create_client(
            GetParameters, "/stretch_driver/get_parameters", callback_group=group
        )
        self._split_moves = True
        self._detect_srv = self.create_client(
            Detect, cfg.detect_service, callback_group=group
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

    # ---- action plumbing ----

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
        try:
            point = self._run(goal_handle, cfg)
            result.success = True
            if cfg.dry_run:
                result.message = "dry_run: plan ok"
            elif cfg.use_gripper:
                result.message = "grasped"
            else:
                result.message = "motion done (use_gripper is false, gripper not used)"
            result.grasped_point = point
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
            MotionError,
            PlanningError,
            SceneError,
            SnapshotError,
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

    # ---- grasping ----

    def _run(self, goal_handle, cfg):
        goal = goal_handle.request
        dry_run = cfg.dry_run

        def cancel():
            return goal_handle.is_cancel_requested

        def phase(p):
            if cancel():
                raise Canceled()
            goal_handle.publish_feedback(Grasp.Feedback(phase=p))

        arm_solver, base_solver = self._solvers(cfg)
        self._split_moves = self._driver_needs_split(cfg, cancel)
        start_pose = cfg.start_pose

        # pre-grasp pose
        phase(Grasp.Feedback.MOVING_TO_START)
        if dry_run:
            current = start_pose
        else:
            self._move(cfg, self._robot.arm_joints(), [start_pose], cancel)
            self._sleep(cfg.settle_time, cancel)
            current = self._robot.arm_joints()

        # detect (& project to 3D)
        phase(Grasp.Feedback.DETECTING)
        grasp_wrist = cfg.grasp_wrist
        rotation = fixed_grasp_rotation(arm_solver, grasp_wrist)
        if goal.mode == Grasp.Goal.MODE_POINT:
            center = self._scene.to_base(
                goal.target_point.header, goal.target_point.point
            )
            self._show_target(cfg, center)
        else:
            center = self._detect_target(cfg, goal.prompt, rotation, cancel)

        # planning: the arm alone if it can, else the base moves first
        plan = self._plan(cfg, arm_solver, base_solver, current, center, grasp_wrist)
        if dry_run:
            self._show_ghost([plan.pregrasp] + plan.approach + [plan.lift], cancel)
            return self._grasped_point(cfg, center)

        # approach
        phase(Grasp.Feedback.APPROACHING)
        base_moves = 0
        while any(abs(v) > 1e-3 for v in plan.base.values()):
            if base_moves == cfg.max_base_moves:
                raise GraspFailed(
                    f"still out of arm reach after {base_moves} base moves"
                )
            self._robot.move_base(
                plan.base.get("base_x", 0.0),
                plan.base.get("base_y", 0.0),
                plan.base.get("base_theta", 0.0),
                cancel,
                BaseGains(
                    max_linear=cfg.base_max_linear,
                    max_angular=cfg.base_max_angular,
                ),
            )
            base_moves += 1
            # TODO[lnfu]
            self._sleep(cfg.settle_time, cancel)
            current = self._robot.arm_joints()
            phase(Grasp.Feedback.DETECTING)
            if (
                goal.mode == Grasp.Goal.MODE_POINT
            ):  # no camera: trust the planned base motion
                moved = base_solver.base_motion({**plan.pregrasp, **plan.base})
                center = moved.actInv(pin.SE3(np.eye(3), center)).translation
                self._show_target(cfg, center)
            else:
                center = self._detect_target(cfg, goal.prompt, rotation, cancel)
            plan = self._plan(
                cfg, arm_solver, base_solver, current, center, grasp_wrist
            )
            phase(Grasp.Feedback.APPROACHING)

        if cfg.use_gripper:
            self._robot.open_gripper(cancel)
        else:
            self.get_logger().warn("use_gripper is false: not opening the gripper")

        self._move(cfg, current, [plan.pregrasp] + plan.approach, cancel)

        # close gripper
        phase(Grasp.Feedback.GRASPING)
        if cfg.use_gripper:
            self._robot.close_gripper(cancel)
            # TODO[lnfu] check the direction once the gripper range is known.
            if self._robot.gripper_position() <= cfg.closed_threshold:
                raise GraspFailed("gripper closed completely: nothing grasped")
        else:
            self.get_logger().warn("use_gripper is false: not closing the gripper")

        # lift
        phase(Grasp.Feedback.LIFTING)
        self._move(cfg, plan.approach[-1], [plan.lift], cancel)
        return self._grasped_point(cfg, center)

    def _detect_target(self, cfg, prompt, rotation, cancel):
        center = self._detect(cfg, prompt, rotation, cancel)
        self._show_target(cfg, center)
        return center

    def _show_target(self, cfg, center):
        # Shown before planning, so it is visible even if IK fails.
        self._scene.publish_markers(center)
        self.get_logger().info(f"target in {cfg.base_frame}: {center.round(3)}")

    def _plan(self, cfg, arm_solver, base_solver, current, center, grasp_wrist):
        """plan_grasp with the configured parameters; base_solver None locks the base."""
        rotation = fixed_grasp_rotation(arm_solver, grasp_wrist)
        plan = plan_grasp(
            arm_solver,
            current,
            center,
            rotation,
            pregrasp_distance=cfg.pregrasp_distance,
            n_waypoints=cfg.n_waypoints,
            lift_height=cfg.lift_height,
            accept_position_error=cfg.ik_accept_position_error,
            accept_orientation_error=np.deg2rad(cfg.ik_accept_orientation_error_deg),
            base_solver=base_solver,
            base_posture={
                "arm_joint": cfg.base_stage_arm,
                "wrist_yaw_joint": grasp_wrist["wrist_yaw_joint"],
            },
        )
        self._scene.publish_markers(
            center, plan.poses["approach"] + [plan.poses["lift"]]
        )
        self.get_logger().info(
            f"plan ok: base {plan.base or 'locked'}, center {center.round(3)}"
        )
        return plan

    def _grasped_point(self, cfg, center):
        """center as a PointStamped in the (current) base frame."""
        grasped = PointStamped()
        grasped.header.frame_id = cfg.base_frame
        grasped.header.stamp = self.get_clock().now().to_msg()
        grasped.point.x, grasped.point.y, grasped.point.z = (float(v) for v in center)
        return grasped

    def _solvers(self, cfg):
        urdf = self._robot.urdf
        if urdf is None:
            raise GraspFailed("no /robot_description received yet")
        try:
            return make_solvers(
                urdf,
                cfg.base_dof_list,
                free_yaw=cfg.free_yaw,
                grasp_yaw=cfg.grasp_wrist["wrist_yaw_joint"],
                max_aim_yaw=np.deg2rad(cfg.max_aim_yaw_deg),
                max_base_translation=cfg.max_base_translation,
                max_base_rotation=np.deg2rad(cfg.max_base_rotation_deg),
                ee_frame=cfg.ee_frame,
                base_frame=cfg.base_frame,
            )
        except ValueError as e:
            raise GraspFailed(f"cannot build IK model: {e}") from e

    def _driver_needs_split(self, cfg, cancel):
        """Whether long moves must be cut up: stretch_driver gives each trajectory point
        action_timeout seconds (3 by default). With a long timeout, e.g. after
        `ros2 param set /stretch_driver action_timeout 15.0`, each move is one point."""
        try:
            res = self._call(
                self._driver_params,
                GetParameters.Request(names=["action_timeout"]),
                cfg.service_timeout,
                cancel,
            )
            timeout = res.values[0].double_value
        except (GraspFailed, IndexError) as e:
            self.get_logger().warn(f"cannot read stretch_driver action_timeout: {e}")
            return True
        split = timeout < cfg.no_split_action_timeout
        self.get_logger().info(
            f"stretch_driver action_timeout {timeout:.1f} s: "
            f"{'splitting long moves' if split else 'one point per move'}"
        )
        return split

    def _move(self, cfg, start, points, cancel):
        if self._split_moves:
            points = split_long_moves(start, points, cfg.max_step)
        self._robot.move_arm(start, points, cancel)

    def _detect(self, cfg, prompt, rotation, cancel):
        """Detect `prompt` (arm already still) and return its center in base_link.

        One color + depth pair is captured here; only the color image goes to the
        GPU PC, and the mask comes back to be combined with the depth kept here.
        """
        snap = take_snapshot(
            self,
            cfg.color_topic,
            cfg.depth_topic,
            cfg.camera_info_topic,
            depth_compressed=cfg.depth_compressed,
            slop=cfg.sync_slop,
            timeout=cfg.snapshot_timeout,
            should_cancel=cancel,
            callback_group=self._group,
        )
        if snap is None:
            raise Canceled()
        color, depth, info = snap

        res = self._call(
            self._detect_srv,
            Detect.Request(prompt=prompt, image=color),
            cfg.service_timeout,
            cancel,
        )
        if not res.success:
            raise GraspFailed(f"detect failed: {res.message}")
        if not res.detections:
            raise GraspFailed(f'no "{prompt}" found')
        best = res.detections[0]  # sorted by score
        self.get_logger().info(f"detected {best.label} ({best.score:.2f})")

        mask = decode_mask_png(best.mask.data)
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            self.get_logger().warn(
                f"{best.label} touches the image edge: it may be cut off, "
                "so its center is uncertain"
            )
        K = np.array(info.k).reshape(3, 3)
        try:
            points = mask_points(
                mask,
                depth_to_array(depth),
                K,
                depth_band=cfg.depth_band,
            )
        except ValueError as e:
            raise GraspFailed(f"{best.label}: {e}") from e

        points = transform_points(points, *self._scene.lookup(color.header))

        width = object_width(points, rotation)
        self.get_logger().info(
            f"{best.label}: {len(points)} points, width {width * 100:.1f} cm"
        )
        if width > cfg.gripper_max_opening:
            raise GraspFailed(
                f"object is {width * 100:.1f} cm wide, wider than the gripper opening"
            )
        return center_and_width(points)[0]

    def _call(self, client, request, timeout, cancel):
        if not client.wait_for_service(timeout_sec=timeout):
            raise GraspFailed(f"service {client.srv_name} not available")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if cancel():
                raise Canceled()
            if time.monotonic() > deadline:
                raise GraspFailed(f"service {client.srv_name} timed out")
            time.sleep(0.02)
        return future.result()

    def _sleep(self, seconds, cancel):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if cancel():
                raise Canceled()
            time.sleep(0.02)

    def _show_ghost(self, waypoints, cancel):
        """Play the arm waypoints on ~/ghost_joint_states, using URDF joint names."""
        for joints in waypoints:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name, msg.position = urdf_joint_positions(joints)
            self._ghost_pub.publish(msg)
            self._sleep(0.3, cancel)


def main():
    rclpy.init()
    node = GraspServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
