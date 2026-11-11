import cv2
import numpy as np

from ...geometry.plane import fit_plane, intersect_plane
from ...geometry.points import depth_to_meters, mask_points, pixel_rays
from .door import CORNERS, Door
from .keypoints import DoorKeypoints


def quad_mask(corners: dict, shape, shrink: float = 0.15) -> np.ndarray:
    quad = np.array(
        [corners[n] for n in ("top_left", "top_right", "bottom_right", "bottom_left")]
    )
    center = quad.mean(axis=0)
    quad = center + (1.0 - shrink) * (quad - center)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
    return mask.astype(bool)


def measure_handle(
    pixel,
    depth: np.ndarray,
    K: np.ndarray,
    normal: np.ndarray,
    d: float,
    min_protrusion: float = 0.01,
    half: int = 4,
) -> np.ndarray | None:
    """Median of the depth points near `pixel` in front of the door. None if none."""
    u, v = round(pixel[0]), round(pixel[1])
    v0, u0 = max(v - half, 0), max(u - half, 0)
    z = depth_to_meters(depth[v0 : v + half + 1, u0 : u + half + 1])
    rows, cols = np.nonzero(np.isfinite(z) & (z > 0))
    if rows.size == 0:
        return None
    zs = z[rows, cols]
    points = pixel_rays(np.column_stack((cols + u0, rows + v0)), K) * zs[:, None]
    handle = points[points @ normal - d >= min_protrusion]
    if len(handle) == 0:
        return None
    return np.median(handle, axis=0)


def locate_door(
    keypoints: DoorKeypoints,
    depth: np.ndarray,
    K: np.ndarray,
    depth_band: float = 0.15,
    shrink: float = 0.15,
) -> Door:
    """`keypoints` are pixels of `depth`. Returns the door in the camera frame."""
    mask = quad_mask(keypoints.corners, depth.shape, shrink=shrink)
    points = mask_points(mask, depth, K, depth_band=depth_band)
    normal, d = fit_plane(points)
    pixels = np.array([keypoints.handle] + [keypoints.corners[n] for n in CORNERS])
    on_plane = intersect_plane(pixel_rays(pixels, K), normal, d)
    measured = measure_handle(keypoints.handle, depth, K, normal, d)
    return Door(
        keypoints.type,
        keypoints.handle_type,
        on_plane[0] if measured is None else measured,
        dict(zip(CORNERS, on_plane[1:], strict=True)),
        normal,
        handle_from_depth=measured is not None,
    )
