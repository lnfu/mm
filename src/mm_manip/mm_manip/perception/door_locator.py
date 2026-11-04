"""Door perception on the robot: snapshot -> keypoints (Gemini) -> door plane -> 3D."""

import time

import cv2
import numpy as np

from ..errors import Canceled
from ..scene import SceneError
from .door import Door, KeypointSource, locate_door
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
        callback_group=None,
    ):
        self._node = node
        self._scene = scene
        self._topics = (color_topic, depth_topic, camera_info_topic)
        self._depth_compressed = depth_compressed
        self._sync_slop = sync_slop
        self._snapshot_timeout = snapshot_timeout
        self._group = callback_group

    def locate(
        self,
        prompt: str,
        source: KeypointSource,
        should_cancel,
        depth_band=0.15,
        shrink=0.15,
    ) -> Door:
        """The door `prompt` describes, in the base frame.

        Call it with the camera still. Logs how long each stage took. Raises
        PerceptionError (no camera data, no keypoints, no depth on the door, TF
        missing) or Canceled.
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

        try:
            door = locate_door(
                keypoints,
                depth,
                np.array(info.k).reshape(3, 3),
                depth_band=depth_band,
                shrink=shrink,
            )
            door = door.transform(*self._scene.lookup(color_msg.header))
        except (ValueError, SceneError) as e:
            raise PerceptionError(f"door: {e}") from e
        t3 = time.monotonic()

        self._node.get_logger().info(
            f"door {door.type.name}: snapshot {t1 - t0:.3f} s, "
            f"keypoints {t2 - t1:.3f} s, plane {t3 - t2:.3f} s"
        )
        return door
