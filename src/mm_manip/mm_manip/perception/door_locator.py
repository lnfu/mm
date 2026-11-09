"""Door perception on the robot: snapshot -> keypoints (Gemini) -> door plane -> 3D."""

import os
import time

import cv2
import numpy as np
import rclpy.time
from rclpy.logging import get_logging_directory
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header

from ..errors import Canceled
from ..scene import SceneError
from .door import (
    Door,
    KeypointSource,
    hinge_edge_problem,
    locate_door,
)
from .door_debug import draw_door_keypoints
from .gemini import GeminiError
from .images import depth_to_array
from .locator import PerceptionError
from .snapshot import SnapshotError, take_snapshot


class DoorLocator:
    def __init__(
        self,
        node,
        scene,
        color_topic,
        depth_topic,
        camera_info_topic,
        depth_compressed=True,
        sync_slop=0.02,
        snapshot_timeout=2.0,
        ee_frame="grasp_center_link",
        callback_group=None,
    ):
        self._node = node
        self._ee_frame = ee_frame
        self._scene = scene
        self._topics = (color_topic, depth_topic, camera_info_topic)
        self._depth_compressed = depth_compressed
        self._sync_slop = sync_slop
        self._snapshot_timeout = snapshot_timeout
        self._group = callback_group
        self._debug_pub = node.create_publisher(
            CompressedImage, "~/door_debug/compressed", 1
        )

    def locate(
        self,
        prompt: str,
        source: KeypointSource,
        should_cancel,
        depth_band=0.15,
        shrink=0.15,
        debug_dir="",
    ) -> Door:
        """The door `prompt` describes, in the base frame.

        Call it with the camera still. Logs how long each stage took. Raises
        PerceptionError (no camera data, no keypoints, the hinge edge out of view
        (see hinge_edge_problem), no depth on the door, TF missing) or Canceled.
        """
        t0 = time.monotonic()
        try:
            snap = take_snapshot(
                self._node,
                *self._topics,
                depth_compressed=self._depth_compressed,
                slop=self._sync_slop,
                timeout=self._snapshot_timeout,
                should_cancel=should_cancel,
                callback_group=self._group,
            )
        except SnapshotError as e:
            raise PerceptionError(str(e)) from e
        if snap is None:
            raise Canceled()
        color_msg, depth_msg, info = snap
        # Now, not after Gemini: the TF buffer keeps only the last ~10 s.
        try:
            camera_to_base = self._scene.lookup(color_msg.header)
        except SceneError as e:
            raise PerceptionError(f"camera pose: {e}") from e
        K = np.array(info.k).reshape(3, 3)
        ee_pixel = self._project_ee(color_msg.header, camera_to_base, K)
        age = (
            self._node.get_clock().now()
            - rclpy.time.Time.from_msg(color_msg.header.stamp)
        ).nanoseconds * 1e-9
        color = cv2.imdecode(
            np.frombuffer(bytes(color_msg.data), np.uint8), cv2.IMREAD_COLOR
        )
        if color is None:
            raise PerceptionError(f"cannot decode the color image ({color_msg.format})")
        depth = depth_to_array(depth_msg)
        t1 = time.monotonic()

        try:
            keypoints = source.keypoints(
                color, prompt, (depth.shape[1], depth.shape[0]), should_cancel
            )
        except (GeminiError, ValueError) as e:
            raise PerceptionError(f"keypoints: {e}") from e
        t2 = time.monotonic()
        self._debug(
            color, depth, keypoints, shrink, source, color_msg, debug_dir, ee_pixel
        )
        problem = hinge_edge_problem(keypoints)
        if problem is not None:
            raise PerceptionError(problem)

        try:
            door = locate_door(
                keypoints, depth, K, depth_band=depth_band, shrink=shrink
            )
            door = door.transform(*camera_to_base)
        except ValueError as e:
            raise PerceptionError(f"door: {e}") from e
        t3 = time.monotonic()

        self._node.get_logger().info(
            f"door {door.type.name}, handle from "
            f"{'depth' if door.handle_from_depth else 'the door plane'}: "
            f"snapshot {t1 - t0:.3f} s (image {age:.2f} s old), "
            f"keypoints {t2 - t1:.3f} s, plane {t3 - t2:.3f} s"
        )
        return door

    def _project_ee(self, header, camera_to_base, K):
        """Pixel (in the depth image) where TF puts the ee frame (between the
        fingertips, which the camera sees), or None. Checks the camera-to-gripper
        calibration: it should land between the fingertips in the image."""
        try:
            ee_to_base = self._scene.lookup(
                Header(frame_id=self._ee_frame, stamp=header.stamp)
            )
        except SceneError as e:
            self._node.get_logger().warning(f"{self._ee_frame} projection: {e}")
            return None
        rotation, translation = camera_to_base
        point = np.asarray(rotation).T @ (ee_to_base[1] - np.asarray(translation))
        if point[2] <= 0:
            self._node.get_logger().info(f"{self._ee_frame} is behind the camera")
            return None
        u, v, _ = K @ point / point[2]
        self._node.get_logger().info(
            f"{self._ee_frame} in the camera (m): {np.round(point, 3).tolist()}, "
            f"at pixel ({u:.0f}, {v:.0f})"
        )
        return (float(u), float(v))

    def _debug(
        self, color, depth, keypoints, shrink, source, color_msg, debug_dir, ee_pixel
    ):
        """Publish and save the keypoints drawn on color and depth.

        Saved in debug_dir, or "" for <ROS log dir>/door_debug (ROS_LOG_DIR, else
        ~/.ros/log).
        """
        logger = self._node.get_logger()
        answer = getattr(source, "last_answer", None)
        if answer is not None:
            logger.info(f"gemini answer: {answer}")
        try:
            image = draw_door_keypoints(color, depth, keypoints, shrink, ee_pixel)
        except Exception as e:  # noqa: BLE001  debugging must not fail the goal
            logger.warning(f"door debug image: {e}")
            return
        ok, jpeg = cv2.imencode(".jpg", image)
        if ok:
            msg = CompressedImage(header=color_msg.header, format="jpeg")
            msg.data = jpeg.tobytes()
            self._debug_pub.publish(msg)
        debug_dir = debug_dir or os.path.join(get_logging_directory(), "door_debug")
        os.makedirs(debug_dir, exist_ok=True)
        stem = os.path.join(debug_dir, time.strftime("door_%Y%m%d_%H%M%S"))
        cv2.imwrite(stem + ".png", image)
        cv2.imwrite(stem + "_color.png", color)
        if answer is not None:
            with open(stem + ".txt", "w") as f:
                f.write(answer)
        logger.info(f"door debug saved: {stem}.png")
