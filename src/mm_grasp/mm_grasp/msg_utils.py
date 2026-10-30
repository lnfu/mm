"""ROS message <-> numpy helpers."""

import cv2
import numpy as np
from mm_ik.model import ARM_SEGMENTS

_DEPTH_DTYPES = {"16UC1": np.uint16, "mono16": np.uint16, "32FC1": np.float32}


def image_to_array(msg) -> np.ndarray:
    """sensor_msgs/Image (16UC1 / mono16 / 32FC1) -> (H, W) array.

    Handles row padding and byte order.
    """
    if msg.encoding not in _DEPTH_DTYPES:
        raise ValueError(f"unsupported depth encoding {msg.encoding}")
    dtype = np.dtype(_DEPTH_DTYPES[msg.encoding]).newbyteorder(
        ">" if msg.is_bigendian else "<"
    )
    rows = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    rows = rows[:, : msg.width * dtype.itemsize].copy()
    return (
        rows.view(dtype).reshape(msg.height, msg.width).astype(dtype.newbyteorder("="))
    )


def compressed_depth_to_array(msg) -> np.ndarray:
    """sensor_msgs/CompressedImage from a compressedDepth topic -> (H, W) uint16 (mm).

    Payload is compressed_depth_image_transport's 12-byte ConfigHeader followed by
    a 16-bit PNG. The format string may carry extra tokens, e.g.
    "16UC1; compressedDepth png; seq=3989" from the Stretch driver.
    """
    tokens = [t.strip() for t in msg.format.split(";")]
    if len(tokens) < 2 or tokens[0] != "16UC1" or tokens[1] != "compressedDepth png":
        raise ValueError(
            f'unsupported compressedDepth format "{msg.format}" (need 16UC1 png)'
        )
    depth = cv2.imdecode(
        np.frombuffer(bytes(msg.data), np.uint8)[12:], cv2.IMREAD_UNCHANGED
    )
    if depth is None or depth.dtype != np.uint16:
        raise ValueError("cannot decode compressedDepth PNG")
    return depth


def depth_to_array(msg) -> np.ndarray:
    """Depth from either a raw Image or a compressedDepth CompressedImage."""
    return (
        compressed_depth_to_array(msg)
        if hasattr(msg, "format")
        else image_to_array(msg)
    )


def decode_mask_png(data) -> np.ndarray:
    """mono8 PNG bytes (Detection.mask from mm_perception) -> bool mask."""
    img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cannot decode mask PNG")
    return img > 127


def quaternion_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_to_rt(transform):
    """geometry_msgs/Transform -> (R (3, 3), t (3,))."""
    t = transform.translation
    return quaternion_to_matrix(transform.rotation), np.array([t.x, t.y, t.z])


def urdf_joint_positions(joints: dict):
    """IK joint dict -> (names, positions) with URDF names, e.g. for a ghost JointState.

    arm_joint is spread evenly over arm_l1..arm_l4, the way stretch_driver publishes it.
    """
    names, positions = [], []
    for name, value in joints.items():
        if name == "arm_joint":
            names += ARM_SEGMENTS
            positions += [value / len(ARM_SEGMENTS)] * len(ARM_SEGMENTS)
        else:
            names.append(name)
            positions.append(float(value))
    return names, positions


def joint_trajectory(names: list, points: list, times: list):
    """Build a trajectory_msgs/JointTrajectory. `points` are dicts keyed by `names`."""
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    traj = JointTrajectory(joint_names=list(names))
    for p, t in zip(points, times):
        sec = int(t)
        traj.points.append(
            JointTrajectoryPoint(
                positions=[float(p[n]) for n in names],
                time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9)),
            )
        )
    return traj
