"""Grab one synchronized color + depth + camera_info (from gripper camera)."""

import threading
import time

import message_filters
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class SnapshotError(RuntimeError):
    pass


def take_snapshot(
    node,
    color_topic,
    depth_topic,
    camera_info_topic,
    depth_compressed=True,
    slop=0.02,
    timeout=2.0,
    should_cancel=lambda: False,
    callback_group=None,
):
    """Return (color CompressedImage, depth, CameraInfo) with stamps within `slop` s.

    depth is a CompressedImage (compressedDepth) if depth_compressed, else an Image.

    The first triple to arrive after the call starts is returned. Call it with the
    camera still, so that triple shows the current scene.
    Raises SnapshotError on timeout; returns None if should_cancel() turns true.
    """
    got = threading.Event()
    result = []

    def on_triple(color, depth, info):
        if not got.is_set():
            result.append((color, depth, info))
            got.set()

    subs = [
        message_filters.Subscriber(
            node,
            msg_type,
            topic,
            qos_profile=qos_profile_sensor_data,
            callback_group=callback_group,
        )
        for msg_type, topic in (
            (CompressedImage, color_topic),
            (CompressedImage if depth_compressed else Image, depth_topic),
            (CameraInfo, camera_info_topic),
        )
    ]
    sync = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=slop)
    sync.registerCallback(on_triple)
    try:
        deadline = time.monotonic() + timeout
        while not got.wait(0.02):
            if should_cancel():
                return None
            if time.monotonic() > deadline:
                raise SnapshotError(
                    f"no synchronized {color_topic} + {depth_topic} + {camera_info_topic} "
                    f"within {timeout} s"
                )
    finally:
        for sub in subs:
            node.destroy_subscription(sub.sub)
    return result[0]
