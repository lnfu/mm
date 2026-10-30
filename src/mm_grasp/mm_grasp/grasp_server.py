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
from mm_ik.solver import ARM_JOINT_NAMES
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from mm_interfaces.action import Grasp
from mm_interfaces.srv import Detect

from .base_control import BaseGains
from .localize import center_and_width, mask_points, transform_points
from .msg_utils import decode_mask_png, depth_to_array, urdf_joint_positions
from .plan import (
    DEFAULT_MAX_STEP,
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

WRIST_JOINTS = ["wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]


class GraspFailed(RuntimeError):
    pass


class GraspServer(Node):
    def __init__(self):
        super().__init__("mm_grasp")
        p = self.declare_parameter
        p("dry_run", True)
        p("base_dofs", "")  # comma separated subset of x,y,theta
        p("base_frame", "base_link")
        p("ee_frame", "grasp_center_link")
        p("fjt_action", "/follow_joint_trajectory")  # served by /stretch_driver
        for name in ARM_JOINT_NAMES:
            p(f"driver_joint.{name}", name)  # the driver uses the same names
            p(f"start_pose.{name}", 0.0)
            p(f"max_step.{name}", DEFAULT_MAX_STEP[name])
        for name in WRIST_JOINTS:
            p(f"grasp_wrist.{name}", 0.0)
        p("pregrasp_distance", 0.10)
        p(
            "n_waypoints", 2
        )  # stretch_driver stops at every point; arm-only approach is already straight
        p("lift_height", 0.10)
        # Let wrist yaw aim at objects off the gripper's line (base locked), keeping the
        # gripper level. The approach is still straight along base x.
        p("free_yaw", True)
        p(
            "max_aim_yaw_deg", 10.0
        )  # ~5 cm sideways; anything further is left to the base
        # The base is only used when the arm alone cannot reach (base_dofs not empty).
        # Then it moves so the arm ends near base_stage_arm at the pre-grasp, leaving
        # room for the approach (0.52 max - pregrasp_distance - margin).
        p("base_stage_arm", 0.35)
        p("max_base_translation", 0.6)  # m, per axis
        p("max_base_rotation_deg", 45.0)
        p("base_max_linear", 0.05)  # m/s
        p("base_max_angular", 0.2)  # rad/s
        # A target slightly off the gripper's line (base locked) never converges exactly.
        p("ik_accept_position_error", 0.015)  # m
        p("ik_accept_orientation_error_deg", 3.0)
        # If stretch_driver's action_timeout is at least this long, long moves are not
        # split into max_step pieces (fewer stops).
        p("no_split_action_timeout", 10.0)
        p("settle_time", 0.3)
        p("service_timeout", 5.0)
        # TODO[lnfu] the gripper mechanism is broken; open/close/check are skipped
        # while use_gripper is false and the arm motions still run.
        p("use_gripper", False)
        p("gripper_max_opening", 0.10)  # TODO[lnfu] measure
        p("closed_threshold", 0.0)  # TODO[lnfu] from the empty vs holding test
        p("detect_service", "/mm_perception/detect")
        # The driver aligns depth to the right camera (setDepthAlign(right)) and
        # publishes it only as compressedDepth, so the color image is the right one.
        p("color_topic", "/cameras_gripper/right/image_raw/compressed")
        p("depth_topic", "/cameras_gripper/stereo/image_raw/compressedDepth")
        p("depth_compressed", True)
        p("camera_info_topic", "/cameras_gripper/right/camera_info")
        p("sync_slop", 0.02)
        p("snapshot_timeout", 2.0)
        p("depth_band", 0.05)

        group = ReentrantCallbackGroup()
        driver_names = {n: self._param(f"driver_joint.{n}") for n in ARM_JOINT_NAMES}
        self._robot = Robot(
            self, self._param("fjt_action"), driver_names, callback_group=group
        )
        self._scene = Scene(self, base_frame=self._param("base_frame"))
        self._group = group
        self._driver_params = self.create_client(
            GetParameters, "/stretch_driver/get_parameters", callback_group=group
        )
        self._split_moves = True
        self._detect_srv = self.create_client(
            Detect, self._param("detect_service"), callback_group=group
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
        self.get_logger().info(f"READY (dry_run={self._param('dry_run')})")

    def _param(self, name):
        return self.get_parameter(name).value

    def _joint_params(self, prefix, names=ARM_JOINT_NAMES):
        return {n: float(self._param(f"{prefix}.{n}")) for n in names}

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
        try:
            point = self._run(goal_handle)
            result.success = True
            if self._param("dry_run"):
                result.message = "dry_run: plan ok"
            elif self._param("use_gripper"):
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

    def _run(self, goal_handle):
        goal = goal_handle.request
        dry_run = self._param("dry_run")
        cancel = lambda: goal_handle.is_cancel_requested

        def phase(p):
            if cancel():
                raise Canceled()
            goal_handle.publish_feedback(Grasp.Feedback(phase=p))

        arm_solver, base_solver = self._solvers()
        self._split_moves = self._driver_needs_split(cancel)
        start_pose = self._joint_params("start_pose")

        # pre-grasp pose
        phase(Grasp.Feedback.MOVING_TO_START)
        if dry_run:
            current = start_pose
        else:
            self._move(self._robot.arm_joints(), [start_pose], cancel)
            self._sleep(self._param("settle_time"), cancel)
            current = self._robot.arm_joints()

        # detect (& project to 3D)
        phase(Grasp.Feedback.DETECTING)
        grasp_wrist = self._joint_params("grasp_wrist", WRIST_JOINTS)
        rotation = fixed_grasp_rotation(arm_solver, grasp_wrist)
        if goal.mode == Grasp.Goal.MODE_POINT:
            center = self._scene.to_base(
                goal.target_point.header, goal.target_point.point
            )
            self._show_target(center)
        else:
            center = self._detect_target(goal.prompt, rotation, cancel)

        # planning: the arm alone if it can, else the base moves first
        plan = self._plan(arm_solver, base_solver, current, center, grasp_wrist)
        if dry_run:
            self._show_ghost([plan.pregrasp] + plan.approach + [plan.lift], cancel)
            return self._grasped_point(center)

        # approach
        phase(Grasp.Feedback.APPROACHING)
        if any(abs(v) > 1e-3 for v in plan.base.values()):
            self._robot.move_base(
                plan.base.get("base_x", 0.0),
                plan.base.get("base_y", 0.0),
                plan.base.get("base_theta", 0.0),
                cancel,
                BaseGains(
                    max_linear=self._param("base_max_linear"),
                    max_angular=self._param("base_max_angular"),
                ),
            )
            # TODO[lnfu]
            self._sleep(self._param("settle_time"), cancel)
            current = self._robot.arm_joints()
            phase(Grasp.Feedback.DETECTING)
            if (
                goal.mode == Grasp.Goal.MODE_POINT
            ):  # no camera: trust the planned base motion
                moved = base_solver.base_motion({**plan.pregrasp, **plan.base})
                center = moved.actInv(pin.SE3(np.eye(3), center)).translation
                self._show_target(center)
            else:
                center = self._detect_target(goal.prompt, rotation, cancel)
            plan = self._plan(arm_solver, None, current, center, grasp_wrist)
            phase(Grasp.Feedback.APPROACHING)

        if self._param("use_gripper"):
            self._robot.open_gripper(cancel)
        else:
            self.get_logger().warn("use_gripper is false: not opening the gripper")

        self._move(current, [plan.pregrasp] + plan.approach, cancel)

        # close gripper
        phase(Grasp.Feedback.GRASPING)
        if self._param("use_gripper"):
            self._robot.close_gripper(cancel)
            # TODO[lnfu] check the direction once the gripper range is known.
            if self._robot.gripper_position() <= self._param("closed_threshold"):
                raise GraspFailed("gripper closed completely: nothing grasped")
        else:
            self.get_logger().warn("use_gripper is false: not closing the gripper")

        # lift
        phase(Grasp.Feedback.LIFTING)
        self._move(plan.approach[-1], [plan.lift], cancel)
        return self._grasped_point(center)

    def _detect_target(self, prompt, rotation, cancel):
        center = self._detect(prompt, rotation, cancel)
        self._show_target(center)
        return center

    def _show_target(self, center):
        # Shown before planning, so it is visible even if IK fails.
        self._scene.publish_markers(center)
        self.get_logger().info(
            f"target in {self._param('base_frame')}: {center.round(3)}"
        )

    def _plan(self, arm_solver, base_solver, current, center, grasp_wrist):
        """plan_grasp with the configured parameters; base_solver None locks the base."""
        rotation = fixed_grasp_rotation(arm_solver, grasp_wrist)
        plan = plan_grasp(
            arm_solver,
            current,
            center,
            rotation,
            pregrasp_distance=self._param("pregrasp_distance"),
            n_waypoints=self._param("n_waypoints"),
            lift_height=self._param("lift_height"),
            accept_position_error=self._param("ik_accept_position_error"),
            accept_orientation_error=np.deg2rad(
                self._param("ik_accept_orientation_error_deg")
            ),
            base_solver=base_solver,
            base_posture={
                "arm_joint": self._param("base_stage_arm"),
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

    def _grasped_point(self, center):
        """center as a PointStamped in the (current) base frame."""
        grasped = PointStamped()
        grasped.header.frame_id = self._param("base_frame")
        grasped.header.stamp = self.get_clock().now().to_msg()
        grasped.point.x, grasped.point.y, grasped.point.z = (float(v) for v in center)
        return grasped

    def _solvers(self):
        urdf = self._robot.urdf
        if urdf is None:
            raise GraspFailed("no /robot_description received yet")
        base_dofs = [
            d.strip() for d in self._param("base_dofs").split(",") if d.strip()
        ]
        try:
            return make_solvers(
                urdf,
                base_dofs,
                free_yaw=self._param("free_yaw"),
                grasp_yaw=self._param("grasp_wrist.wrist_yaw_joint"),
                max_aim_yaw=np.deg2rad(self._param("max_aim_yaw_deg")),
                max_base_translation=self._param("max_base_translation"),
                max_base_rotation=np.deg2rad(self._param("max_base_rotation_deg")),
                ee_frame=self._param("ee_frame"),
                base_frame=self._param("base_frame"),
            )
        except ValueError as e:
            raise GraspFailed(f"cannot build IK model: {e}") from e

    def _driver_needs_split(self, cancel):
        """Whether long moves must be cut up: stretch_driver gives each trajectory point
        action_timeout seconds (3 by default). With a long timeout, e.g. after
        `ros2 param set /stretch_driver action_timeout 15.0`, each move is one point."""
        try:
            res = self._call(
                self._driver_params,
                GetParameters.Request(names=["action_timeout"]),
                cancel,
            )
            timeout = res.values[0].double_value
        except (GraspFailed, IndexError) as e:
            self.get_logger().warn(f"cannot read stretch_driver action_timeout: {e}")
            return True
        split = timeout < self._param("no_split_action_timeout")
        self.get_logger().info(
            f"stretch_driver action_timeout {timeout:.1f} s: "
            f"{'splitting long moves' if split else 'one point per move'}"
        )
        return split

    def _move(self, start, points, cancel):
        if self._split_moves:
            points = split_long_moves(start, points, self._joint_params("max_step"))
        self._robot.move_arm(start, points, cancel)

    def _detect(self, prompt, rotation, cancel):
        """Detect `prompt` (arm already still) and return its center in base_link.

        One color + depth pair is captured here; only the color image goes to the
        GPU PC, and the mask comes back to be combined with the depth kept here.
        """
        snap = take_snapshot(
            self,
            self._param("color_topic"),
            self._param("depth_topic"),
            self._param("camera_info_topic"),
            depth_compressed=self._param("depth_compressed"),
            slop=self._param("sync_slop"),
            timeout=self._param("snapshot_timeout"),
            should_cancel=cancel,
            callback_group=self._group,
        )
        if snap is None:
            raise Canceled()
        color, depth, info = snap

        res = self._call(
            self._detect_srv, Detect.Request(prompt=prompt, image=color), cancel
        )
        if not res.success:
            raise GraspFailed(f"detect failed: {res.message}")
        if not res.detections:
            raise GraspFailed(f'no "{prompt}" found')
        best = res.detections[0]  # sorted by score
        self.get_logger().info(f"detected {best.label} ({best.score:.2f})")

        K = np.array(info.k).reshape(3, 3)
        try:
            points = mask_points(
                decode_mask_png(best.mask.data),
                depth_to_array(depth),
                K,
                depth_band=self._param("depth_band"),
            )
        except ValueError as e:
            raise GraspFailed(f"{best.label}: {e}") from e

        points = transform_points(points, *self._scene.lookup(color.header))

        width = object_width(points, rotation)
        self.get_logger().info(
            f"{best.label}: {len(points)} points, width {width * 100:.1f} cm"
        )
        if width > self._param("gripper_max_opening"):
            raise GraspFailed(
                f"object is {width * 100:.1f} cm wide, wider than the gripper opening"
            )
        return center_and_width(points)[0]

    def _call(self, client, request, cancel):
        timeout = self._param("service_timeout")
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
