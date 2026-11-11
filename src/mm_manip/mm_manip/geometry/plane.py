import numpy as np


def fit_plane(
    points: np.ndarray,
    inlier_distance: float = 0.01,
    max_points: int = 5000,
    max_iterations: int = 500,
    confidence: float = 0.999,
    seed: int = 0,
):
    """RANSAC (MSAC score), then least squares on the inliers.

    Returns (normal, d) with normal.p = d. The normal points to the camera (d < 0).
    """
    if len(points) < 3:
        raise ValueError(f"only {len(points)} points to fit a plane")
    rng = np.random.default_rng(seed)
    sample = (
        points[rng.choice(len(points), max_points, replace=False)]
        if len(points) > max_points
        else points
    )
    threshold2 = inlier_distance**2

    best_cost, best = np.inf, None
    needed, i = max_iterations, 0
    while i < min(needed, max_iterations):
        i += 1
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        distance2 = (sample @ normal - normal @ a) ** 2
        cost = np.minimum(distance2, threshold2).sum()
        if cost < best_cost:
            best_cost, best = cost, (normal, float(normal @ a))
            needed = ransac_iterations(
                (distance2 < threshold2).mean(), confidence, max_iterations
            )
    if best is None:
        raise ValueError("no plane: the points are collinear")

    normal, d = best
    inliers = np.abs(points @ normal - d) < inlier_distance
    if inliers.sum() >= 3:
        normal, d = _least_squares_plane(points[inliers])
    if d > 0:
        normal, d = -normal, -d
    return normal, d


def ransac_iterations(inlier_ratio: float, confidence: float, cap: int) -> int:
    p_good = inlier_ratio**3
    if p_good >= 1.0:
        return 1
    if p_good <= 0.0:
        return cap
    return min(cap, int(np.ceil(np.log(1.0 - confidence) / np.log1p(-p_good))))


def _least_squares_plane(points: np.ndarray):
    centroid = points.mean(axis=0)
    # full_matrices=False: else U is N x N (many GB).
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][-1]
    return normal, float(normal @ centroid)


def intersect_plane(rays: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    denominators = rays @ normal
    if np.any(np.abs(denominators) < 1e-9):
        raise ValueError("a keypoint ray is parallel to the door plane")
    t = d / denominators
    if np.any(t <= 0):
        raise ValueError("a keypoint ray hits the door plane behind the camera")
    return rays * t[:, None]
