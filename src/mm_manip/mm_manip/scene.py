import numpy as np
import rclpy.duration
import rclpy.time
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .errors import SkillError
from .motion.trajectory import urdf_joint_positions
from .transforms import transform_to_rt


class SceneError(SkillError):
    pass


class Scene:
    def __init__(self, node, base_frame="base_link"):
        self._node = node
        self._base_frame = base_frame
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._marker_pub = node.create_publisher(MarkerArray, "~/target_markers", 1)
        self._ghost_pub = node.create_publisher(JointState, "~/ghost_joint_states", 1)

    def lookup(self, header):
        if header.frame_id in ("", self._base_frame):
            return np.eye(3), np.zeros(3)
        stamp = rclpy.time.Time.from_msg(header.stamp)  # zero: the latest TF
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                header.frame_id,
                stamp if stamp.nanoseconds else rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:
            raise SceneError(f"TF {self._base_frame} <- {header.frame_id}: {e}") from e
        return transform_to_rt(tf.transform)

    def to_base(self, header, point) -> np.ndarray:
        R, t = self.lookup(header)
        return R @ np.array([point.x, point.y, point.z]) + t

    def publish_markers(self, center, poses=()):
        stamp = self._node.get_clock().now().to_msg()
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))
        m = self._marker(stamp, 0, Marker.SPHERE, (1.0, 0.3, 0.1))
        m.scale.x = m.scale.y = m.scale.z = 0.03
        m.pose.position = Point(
            x=float(center[0]), y=float(center[1]), z=float(center[2])
        )
        markers.markers.append(m)
        for i, pose in enumerate(poses):
            m = self._marker(stamp, 10 + i, Marker.ARROW, (0.1, 0.9, 0.2))
            m.scale.x, m.scale.y, m.scale.z = 0.003, 0.006, 0.01
            tip = pose.translation
            tail = tip - 0.05 * pose.rotation[:, 0]
            m.points = [
                Point(x=float(a), y=float(b), z=float(c)) for a, b, c in (tail, tip)
            ]
            markers.markers.append(m)
        self._marker_pub.publish(markers)

    def publish_ghost(self, joints: dict):
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name, msg.position = urdf_joint_positions(joints)
        self._ghost_pub.publish(msg)

    def _marker(self, stamp, marker_id, marker_type, rgb):
        m = Marker(id=marker_id, type=marker_type, action=Marker.ADD, ns="mm_manip")
        m.header.frame_id = self._base_frame
        m.header.stamp = stamp
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 1.0)
        return m
