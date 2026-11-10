import threading
import time

from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from mm_ik.joints import ARM_JOINT_NAMES, ARM_SEGMENTS

from ..errors import MotionError
from ..motion.base_control import Pose2D, yaw_from_quaternion


class RobotState:
    """The latest joint states, odometry, URDF and stretch_driver mode."""

    def __init__(self, node, driver_joint_names: dict, callback_group=None):
        self._driver_names = driver_joint_names
        self._lock = threading.Lock()
        self._joints = {}
        self._joints_time = None
        self._odom = None  # (Pose2D, receive time)
        self._urdf = None
        self._mode = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for msg_type, topic, callback, qos in (
            (JointState, "/joint_states", self._on_joint_states, 10),
            (Odometry, "/wheel_odom", self._on_odom, 10),
            (String, "/robot_description", self._on_urdf, latched),
            (String, "/mode", self._on_mode, latched),
        ):
            node.create_subscription(
                msg_type, topic, callback, qos, callback_group=callback_group
            )

    def _on_joint_states(self, msg):
        with self._lock:
            self._joints.update(zip(msg.name, msg.position, strict=False))
            self._joints_time = time.monotonic()

    def _on_odom(self, msg):
        p = msg.pose.pose
        pose = Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        with self._lock:
            self._odom = (pose, time.monotonic())

    def _on_urdf(self, msg):
        self._urdf = msg.data

    def _on_mode(self, msg):
        self._mode = msg.data

    @property
    def urdf(self) -> str | None:
        return self._urdf

    @property
    def mode(self) -> str | None:
        return self._mode

    def arm_joints(self, max_age: float = None) -> dict:
        with self._lock:
            state = dict(self._joints)
            stamp = self._joints_time
        if max_age is not None and (
            stamp is None or time.monotonic() - stamp > max_age
        ):
            raise MotionError(f"/joint_states is older than {max_age} s")
        joints = {}
        for name in ARM_JOINT_NAMES:
            driver = self._driver_names[name]
            if driver in state:
                joints[name] = state[driver]
            # stretch_driver publishes arm_l1..arm_l4, each 1/4 of arm_joint.
            elif name == "arm_joint" and all(s in state for s in ARM_SEGMENTS):
                joints[name] = sum(state[s] for s in ARM_SEGMENTS)
            else:
                raise MotionError(f"{driver} not in /joint_states")
        return joints

    def joint(self, name: str) -> float:
        with self._lock:
            position = self._joints.get(name)
        if position is None:
            raise MotionError(f"{name} not in /joint_states")
        return position

    def odom_pose(self, max_age: float) -> Pose2D:
        with self._lock:
            odom = self._odom
        if odom is None:
            raise MotionError("no /wheel_odom received")
        pose, stamp = odom
        if time.monotonic() - stamp > max_age:
            raise MotionError(f"/wheel_odom is older than {max_age} s")
        return pose
