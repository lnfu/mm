"""Object perception on the robot: snapshot -> Detect (GPU PC) -> 3D points."""

import time
from dataclasses import dataclass

import numpy as np

from mm_interfaces.srv import Detect

from .localize import mask_points, touches_edge, transform_points
from .msg_utils import decode_mask_png, depth_to_array
from .robot import Canceled
from .scene import SceneError
from .snapshot import SnapshotError, take_snapshot


class PerceptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    label: str
    score: float
    points: np.ndarray  # (N, 3) in the base frame
    truncated: bool  # the mask reaches the image border: part may be out of view


class Locator:
    def __init__(
        self,
        node,
        scene,
        detect_service,
        color_topic,
        depth_topic,
        camera_info_topic,
        depth_compressed=True,
        sync_slop=0.02,
        snapshot_timeout=2.0,
        service_timeout=5.0,
        depth_band=0.05,
        callback_group=None,
    ):
        self._node = node
        self._scene = scene
        self._topics = (color_topic, depth_topic, camera_info_topic)
        self._depth_compressed = depth_compressed
        self._sync_slop = sync_slop
        self._snapshot_timeout = snapshot_timeout
        self._service_timeout = service_timeout
        self._depth_band = depth_band
        self._group = callback_group
        self._detect = node.create_client(
            Detect, detect_service, callback_group=callback_group
        )

    def locate(self, prompt: str, should_cancel) -> Observation:
        """The best-scoring `prompt` in view, as points in the base frame.

        Call it with the camera still. Raises PerceptionError (nothing found, no
        camera data, no depth on the object, TF missing) or Canceled.
        """
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
        color, depth, info = snap

        res = self._call(Detect.Request(prompt=prompt, image=color), should_cancel)
        if not res.success:
            raise PerceptionError(f"detect failed: {res.message}")
        if not res.detections:
            raise PerceptionError(f'no "{prompt}" found')
        best = res.detections[0]  # sorted by score
        self._node.get_logger().info(f"detected {best.label} ({best.score:.2f})")

        mask = decode_mask_png(best.mask.data)
        truncated = touches_edge(mask)
        if truncated:
            self._node.get_logger().warn(
                f"{best.label} touches the image edge: it may be cut off, "
                "so its position is uncertain"
            )
        try:
            points = mask_points(
                mask,
                depth_to_array(depth),
                np.array(info.k).reshape(3, 3),
                depth_band=self._depth_band,
            )
            points = transform_points(points, *self._scene.lookup(color.header))
        except (ValueError, SceneError) as e:
            raise PerceptionError(f"{best.label}: {e}") from e
        return Observation(best.label, best.score, points, truncated)

    def _call(self, request, should_cancel):
        if not self._detect.wait_for_service(timeout_sec=self._service_timeout):
            raise PerceptionError(f"service {self._detect.srv_name} not available")
        future = self._detect.call_async(request)
        deadline = time.monotonic() + self._service_timeout
        while not future.done():
            if should_cancel():
                raise Canceled()
            if time.monotonic() > deadline:
                raise PerceptionError(f"service {self._detect.srv_name} timed out")
            time.sleep(0.02)
        return future.result()
