import os
import time

import cv2
from rclpy.logging import get_logging_directory
from sensor_msgs.msg import CompressedImage

from ..camera.camera import Camera, header
from ..camera.frame import Frame
from ..cancel import CancelToken
from .door import Door, KeypointSource, hinge_edge_problem, locate_door
from .door_debug import draw_door_keypoints
from .gemini import GeminiError
from .locator import PerceptionError


class DoorLocator:
    def __init__(self, node, camera: Camera, ee_frame="grasp_center_link"):
        self._node = node
        self._camera = camera
        self._ee_frame = ee_frame
        self._debug_pub = node.create_publisher(
            CompressedImage, "~/door_debug/compressed", 1
        )

    def locate(
        self,
        prompt: str,
        source: KeypointSource,
        cancel: CancelToken,
        depth_band=0.15,
        shrink=0.15,
        debug_dir="",
    ) -> Door:
        """The door `prompt` describes, in the base frame. The camera must be still."""
        t0 = time.monotonic()
        frame = self._camera.capture(cancel)
        ee_pixel = self._camera.pixel_of_frame(frame, self._ee_frame)
        self._node.get_logger().info(f"{self._ee_frame} at pixel {ee_pixel}")
        t1 = time.monotonic()

        try:
            keypoints = source.keypoints(frame.color, prompt, frame.depth_size, cancel)
        except (GeminiError, ValueError) as e:
            raise PerceptionError(f"keypoints: {e}") from e
        t2 = time.monotonic()
        self._debug(frame, keypoints, shrink, source, debug_dir, ee_pixel)
        problem = hinge_edge_problem(keypoints)
        if problem is not None:
            raise PerceptionError(problem)

        try:
            door = locate_door(
                keypoints, frame.depth, frame.K, depth_band=depth_band, shrink=shrink
            )
        except ValueError as e:
            raise PerceptionError(f"door: {e}") from e
        door = door.transform(*frame.camera_to_base)
        t3 = time.monotonic()

        self._node.get_logger().info(
            f"door {door.type.name}, handle from "
            f"{'depth' if door.handle_from_depth else 'the door plane'}: "
            f"snapshot {t1 - t0:.3f} s (image {frame.age:.2f} s old), "
            f"keypoints {t2 - t1:.3f} s, plane {t3 - t2:.3f} s"
        )
        return door

    def _debug(self, frame: Frame, keypoints, shrink, source, debug_dir, ee_pixel):
        logger = self._node.get_logger()
        answer = getattr(source, "last_answer", None)
        if answer is not None:
            logger.info(f"gemini answer: {answer}")
        try:
            image = draw_door_keypoints(
                frame.color, frame.depth, keypoints, shrink, ee_pixel
            )
        except Exception as e:  # noqa: BLE001  debugging must not fail the goal
            logger.warning(f"door debug image: {e}")
            return
        ok, jpeg = cv2.imencode(".jpg", image)
        if ok:
            msg = CompressedImage(header=header(frame), format="jpeg")
            msg.data = jpeg.tobytes()
            self._debug_pub.publish(msg)
        debug_dir = debug_dir or os.path.join(get_logging_directory(), "door_debug")
        os.makedirs(debug_dir, exist_ok=True)
        stem = os.path.join(debug_dir, time.strftime("door_%Y%m%d_%H%M%S"))
        cv2.imwrite(stem + ".png", image)
        cv2.imwrite(stem + "_color.png", frame.color)
        if answer is not None:
            with open(stem + ".txt", "w") as f:
                f.write(answer)
        logger.info(f"door debug saved: {stem}.png")
