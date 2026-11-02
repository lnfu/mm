import cv2
import numpy as np

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
