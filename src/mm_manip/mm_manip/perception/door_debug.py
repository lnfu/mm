"""Debug picture of Gemini's door keypoints: color and depth side by side. No ROS."""

import cv2
import numpy as np

from .door import CORNERS, DoorKeypoints, quad_mask

_QUAD = ("top_left", "top_right", "bottom_right", "bottom_left")


def draw_door_keypoints(
    color: np.ndarray,
    depth: np.ndarray,
    keypoints: DoorKeypoints,
    shrink: float,
    ee_pixel=None,
) -> np.ndarray:
    """BGR image: left the color image, right the depth, each with the handle
    (red), the corners (yellow) and the shrunk quad the plane is fitted in (green).
    On the color image, `ee_pixel` (magenta): where TF puts the gripper's grasp
    center; it should sit between the fingertips.

    `keypoints` and `ee_pixel` are pixels of `depth` (as locate_door takes them).
    """
    sx = color.shape[1] / depth.shape[1]
    sy = color.shape[0] / depth.shape[0]

    def on_color(p):
        return (round(p[0] * sx), round(p[1] * sy))

    def on_depth(p):
        return (round(p[0]), round(p[1]))

    left = color.copy()
    valid = depth > 0
    no_depth = (128, 128, 128)  # gray, not black: TURBO's near end is almost black
    right = np.full((*depth.shape, 3), no_depth, np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], (2, 98))
        scaled = np.clip((depth - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        right = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
        right[~valid] = no_depth
    mask = quad_mask(keypoints.corners, depth.shape, shrink=shrink)
    right[mask] = (0.6 * right[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    for image, to_px in ((left, on_color), (right, on_depth)):
        quad = np.array([to_px(keypoints.corners[n]) for n in _QUAD], np.int32)
        cv2.polylines(image, [quad], True, (0, 255, 0), 2)
        for name in CORNERS:
            p = to_px(keypoints.corners[name])
            cv2.circle(image, p, 6, (0, 255, 255), -1)
            cv2.putText(image, name, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        h = to_px(keypoints.handle)
        cv2.drawMarker(image, h, (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
        cv2.circle(image, h, 8, (0, 0, 255), 2)

    if ee_pixel is not None:
        e = on_color(ee_pixel)
        cv2.drawMarker(left, e, (255, 0, 255), cv2.MARKER_DIAMOND, 20, 2)
        cv2.drawMarker(left, e, (255, 0, 255), cv2.MARKER_CROSS, 40, 1)
        cv2.putText(
            left,
            f"ee (TF) {e}",
            (10, left.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    if right.shape[0] != left.shape[0]:
        right = cv2.resize(
            right,
            (round(right.shape[1] * left.shape[0] / right.shape[0]), left.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    out = np.hstack((left, right))

    u, v = on_depth(keypoints.handle)
    handle_depth = (
        depth[v, u] if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1] else None
    )
    label = (
        f"{keypoints.type.name} {keypoints.handle_type.name}  "
        f"edges visible L {keypoints.sides_visible['left']:d} "
        f"R {keypoints.sides_visible['right']:d}  "
        f"handle px ({u}, {v}) depth {handle_depth}"
    )
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
