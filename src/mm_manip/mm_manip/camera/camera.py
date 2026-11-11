import threading

import cv2
import message_filters
import numpy as np
import rclpy.time
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header

from ..cancel import CancelToken
from ..errors import SkillError
from ..scene import SceneError
from .frame import Frame
from .images import depth_to_array


class CameraError(SkillError):
    pass


class Camera:
    def __init__(
        self,
        node,
        scene,
        color_topic,
        depth_topic,
        camera_info_topic,
        depth_compressed=True,
        sync_slop=0.02,
        timeout=2.0,
        callback_group=None,
    ):
        self._node = node
        self._scene = scene
        self._topics = (
            (CompressedImage, color_topic),
            (CompressedImage if depth_compressed else Image, depth_topic),
            (CameraInfo, camera_info_topic),
        )
        self._sync_slop = sync_slop
        self._timeout = timeout
        self._group = callback_group

    def capture(self, cancel: CancelToken) -> Frame:
        """The first synchronized color, depth and camera_info after the call.

        Call it with the camera still.
        """
        color, depth, info = self._snapshot(cancel)
        try:
            camera_to_base = self._scene.lookup(color.header)
        except SceneError as e:
            raise CameraError(f"camera pose: {e}") from e
        image = cv2.imdecode(
            np.frombuffer(bytes(color.data), np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise CameraError(f"cannot decode the color image ({color.format})")
        try:
            depth_array = depth_to_array(depth)
        except ValueError as e:
            raise CameraError(f"depth: {e}") from e
        stamp = rclpy.time.Time.from_msg(color.header.stamp)
        return Frame(
            color=image,
            color_data=bytes(color.data),
            color_format=color.format,
            depth=depth_array,
            K=np.array(info.k).reshape(3, 3),
            camera_to_base=camera_to_base,
            frame_id=color.header.frame_id,
            stamp_ns=stamp.nanoseconds,
            age=(self._node.get_clock().now() - stamp).nanoseconds * 1e-9,
        )

    def _snapshot(self, cancel: CancelToken):
        got = threading.Event()
        result = []

        def on_triple(*msgs):
            if not got.is_set():
                result.append(msgs)
                got.set()

        subs = [
            message_filters.Subscriber(
                self._node,
                msg_type,
                topic,
                qos_profile=qos_profile_sensor_data,
                callback_group=self._group,
            )
            for msg_type, topic in self._topics
        ]
        sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=10, slop=self._sync_slop
        )
        sync.registerCallback(on_triple)
        try:
            if not cancel.wait_until(got.is_set, self._timeout):
                topics = " + ".join(topic for _, topic in self._topics)
                raise CameraError(f"no synchronized {topics} within {self._timeout} s")
        finally:
            for sub in subs:
                self._node.destroy_subscription(sub.sub)
        return result[0]


def header(frame: Frame, frame_id: str = None) -> Header:
    """A header with the stamp of `frame`, in `frame_id` (default: the camera's)."""
    msg = Header(frame_id=frame_id or frame.frame_id)
    msg.stamp = rclpy.time.Time(nanoseconds=frame.stamp_ns).to_msg()
    return msg
