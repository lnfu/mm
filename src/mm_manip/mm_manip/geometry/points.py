import numpy as np


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    if depth.dtype == np.uint16:  # mm (16UC1); float is m (32FC1)
        return depth.astype(np.float32) * 1e-3
    return depth.astype(np.float32)


def touches_edge(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def resize_mask(mask: np.ndarray, shape) -> np.ndarray:
    if mask.shape == tuple(shape):
        return mask
    rows = (np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int)
    cols = (np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int)
    return mask[rows[:, None], cols[None, :]]


def mask_points(mask, depth, K, depth_band=0.05, min_points=30) -> np.ndarray:
    z = depth_to_meters(depth)
    mask = resize_mask(np.asarray(mask, bool), z.shape)
    valid = mask & np.isfinite(z) & (z > 0)
    if valid.sum() < min_points:
        raise ValueError(f"only {int(valid.sum())} valid depth pixels in mask")

    v, u = np.nonzero(valid)
    zs = z[v, u]
    # Far from the median depth: mostly background at the mask edge.
    keep = np.abs(zs - np.median(zs)) < depth_band
    if keep.sum() < min_points:
        raise ValueError(
            f"only {int(keep.sum())} depth pixels within {depth_band} m of the median"
        )
    u, v, zs = u[keep], v[keep], zs[keep]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.column_stack(((u - cx) * zs / fx, (v - cy) * zs / fy, zs))


def transform_points(points: np.ndarray, pose) -> np.ndarray:
    """(N, 3) points through a pin.SE3."""
    return points @ pose.rotation.T + pose.translation


def center_and_width(points: np.ndarray, width_axis: int = 1, percentile: float = 2.0):
    # The camera sees only the front, so the center is near the front surface.
    center = np.median(points, axis=0)
    lo, hi = np.percentile(points[:, width_axis], [percentile, 100 - percentile])
    return center, float(hi - lo)


def pixel_rays(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    pixels = np.atleast_2d(np.asarray(pixels, float))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.column_stack(
        ((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels)))
    )
