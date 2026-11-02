"""Mask + depth -> 3D points."""

import numpy as np


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """uint16 depth is in mm (ROS convention, 16UC1); float depth is already in m (32FC1)."""
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * 1e-3
    return depth.astype(np.float32)


def touches_edge(mask: np.ndarray) -> bool:
    """Whether the mask reaches the image border, i.e. the object may be cut off."""
    return bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def resize_mask(mask: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour resize, for when the detection image and depth differ in resolution."""
    if mask.shape == tuple(shape):
        return mask
    rows = (np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int)
    cols = (np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int)
    return mask[rows[:, None], cols[None, :]]


def mask_points(mask, depth, K, depth_band=0.05, min_points=30) -> np.ndarray:
    """Back-project masked depth pixels to (N, 3) points in the camera optical frame.

    Pixels farther than `depth_band` from the median depth are dropped: these are
    mostly background leaking through the mask edge.

    Raises ValueError when fewer than `min_points` remain.
    """
    z = depth_to_meters(depth)
    mask = resize_mask(np.asarray(mask, bool), z.shape)
    valid = mask & np.isfinite(z) & (z > 0)
    if valid.sum() < min_points:
        raise ValueError(f"only {int(valid.sum())} valid depth pixels in mask")

    v, u = np.nonzero(valid)
    zs = z[v, u]
    keep = np.abs(zs - np.median(zs)) < depth_band
    if keep.sum() < min_points:
        raise ValueError(
            f"only {int(keep.sum())} depth pixels within {depth_band} m of the median"
        )
    u, v, zs = u[keep], v[keep], zs[keep]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.column_stack(((u - cx) * zs / fx, (v - cy) * zs / fy, zs))


def transform_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply p' = R p + t to (N, 3) points."""
    return points @ np.asarray(rotation).T + np.asarray(translation)


def center_and_width(points: np.ndarray, width_axis: int = 1, percentile: float = 2.0):
    """Object center (median of the points) and its extent along `width_axis`.

    The extent uses the [percentile, 100 - percentile] range to ignore stray points.
    Note the camera only sees the front surface, so the center is biased toward it.
    """
    center = np.median(points, axis=0)
    lo, hi = np.percentile(points[:, width_axis], [percentile, 100 - percentile])
    return center, float(hi - lo)
