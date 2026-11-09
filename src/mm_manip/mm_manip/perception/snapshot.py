import threading

import message_filters
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from ..cancel import CancelToken


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
    cancel: CancelToken = None,
    callback_group=None,
):
    """The first synchronized (color, depth, camera_info) after the call.

    Call it with the camera still.
    """
    cancel = cancel or CancelToken()
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
        if not cancel.wait_until(got.is_set, timeout):
            raise SnapshotError(
                f"no synchronized {color_topic} + {depth_topic} + {camera_info_topic} "
                f"within {timeout} s"
            )
    finally:
        for sub in subs:
            node.destroy_subscription(sub.sub)
    return result[0]
