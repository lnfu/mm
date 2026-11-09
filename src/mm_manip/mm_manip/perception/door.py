import json
import re
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Protocol

import cv2
import numpy as np

from .localize import depth_to_meters, mask_points

CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
SIDES = ("left", "right")


class DoorType(IntEnum):
    # Values match TYPE_* in Articulate.action.
    UNKNOWN = 0
    REVOLUTE_LEFT = 1  # hinge on the left, as the robot sees it
    REVOLUTE_RIGHT = 2
    PRISMATIC = 3


class HandleType(IntEnum):
    # Values match HANDLE_* in Articulate.action.
    UNKNOWN = 0
    BAR_VERTICAL = 1
    BAR_HORIZONTAL = 2
    KNOB = 3
    RECESSED = 4


_TYPE_NAMES = {
    "revolute_left": DoorType.REVOLUTE_LEFT,
    "revolute_right": DoorType.REVOLUTE_RIGHT,
    "prismatic": DoorType.PRISMATIC,
    "unknown": DoorType.UNKNOWN,
}


_HANDLE_NAMES = {h.name.lower(): h for h in HandleType}


@dataclass(frozen=True)
class DoorKeypoints:
    type: DoorType
    handle_type: HandleType
    handle: tuple  # (u, v) px
    corners: dict
    sides_visible: dict  # side -> the door edge on that side is fully in view


class KeypointSource(Protocol):
    def keypoints(
        self, image: np.ndarray, prompt: str, size: tuple, cancel
    ) -> DoorKeypoints: ...


class GeminiDoorKeypoints:
    def __init__(self, client, max_side: int = 640, jpeg_quality: int = 90):
        self._client = client
        self._max_side = max_side
        self._jpeg_quality = jpeg_quality
        self.last_answer = None

    def keypoints(self, image, prompt, size, cancel) -> DoorKeypoints:
        text = self._client.ask(
            encode_jpeg(image, self._max_side, self._jpeg_quality),
            "image/jpeg",
            build_prompt(prompt),
            cancel,
        )
        self.last_answer = text
        return parse_response(text, *size)


def encode_jpeg(image: np.ndarray, max_side: int, quality: int = 90) -> bytes:
    scale = max_side / max(image.shape[:2])
    if scale < 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cannot encode the image as JPEG")
    return data.tobytes()


def build_prompt(target: str) -> str:
    return f"""Find {target} in the image.
Return only JSON, with no other text:
{{"type": "revolute_left" | "revolute_right" | "prismatic" | "unknown",
  "handle_type": "bar_vertical" | "bar_horizontal" | "knob" | "recessed" | "unknown",
  "handle": [y, x],
  "top_left": [y, x], "top_right": [y, x],
  "bottom_left": [y, x], "bottom_right": [y, x],
  "left_edge_visible": true | false, "right_edge_visible": true | false}}
type: revolute_left if the door's hinge is on its left side in the image,
revolute_right if on its right side, prismatic for a drawer.
handle_type: bar_vertical or bar_horizontal for a bar handle (by its long axis),
knob for a single protruding knob, recessed for a recessed handle or none.
handle: the center of its handle. top_left ... bottom_right: the corners of the
door's front panel (the moving part), as they appear in the image. If a corner
is outside the image, give the point where that edge of the door leaves it.
left_edge_visible: true only if the whole left vertical edge of the door's
front panel, from its top corner to its bottom corner, is inside the image and
not hidden; false if the image border cuts the door off on its left, or if you
cannot see where the door ends on its left. right_edge_visible: the same for
the right edge. Judge these from the image alone; do not guess.
Points are [y, x], normalized to 0-1000."""


def parse_response(text: str, width: int, height: int) -> DoorKeypoints:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    try:
        data = json.loads(fenced.group(1) if fenced else text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {text[:200]!r}") from e
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {text[:200]!r}")
    door_type = _TYPE_NAMES.get(str(data.get("type", "unknown")).lower())
    if door_type is None:
        raise ValueError(f"unknown door type {data.get('type')!r}")
    handle_type = _HANDLE_NAMES.get(str(data.get("handle_type", "unknown")).lower())
    if handle_type is None:
        raise ValueError(f"unknown handle type {data.get('handle_type')!r}")

    def pixel(name):
        point = data.get(name)
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(c, int | float) for c in point)
        ):
            raise ValueError(f"{name}: expected [y, x], got {point!r}")
        y, x = point  # Gemini gives [y, x] in 0..1000.
        return (x / 1000.0 * width, y / 1000.0 * height)

    def visible(side):
        value = data.get(f"{side}_edge_visible")
        if not isinstance(value, bool):
            raise ValueError(
                f"{side}_edge_visible: expected true or false, got {value!r}"
            )
        return value

    return DoorKeypoints(
        door_type,
        handle_type,
        pixel("handle"),
        {name: pixel(name) for name in CORNERS},
        {side: visible(side) for side in SIDES},
    )


def hinge_side(door_type: DoorType) -> str | None:
    return {DoorType.REVOLUTE_LEFT: "left", DoorType.REVOLUTE_RIGHT: "right"}.get(
        door_type
    )


def hinge_edge_problem(keypoints: DoorKeypoints) -> str | None:
    side = hinge_side(keypoints.type)
    if side is None:
        return None
    if not keypoints.sides_visible[side]:
        return (
            f"the door's {side} (hinge) edge is not wholly in view, so its turning "
            "radius is unknown: move the camera to see the whole door"
        )
    return None


def quad_mask(corners: dict, shape, shrink: float = 0.15) -> np.ndarray:
    quad = np.array(
        [corners[n] for n in ("top_left", "top_right", "bottom_right", "bottom_left")]
    )
    center = quad.mean(axis=0)
    quad = center + (1.0 - shrink) * (quad - center)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
    return mask.astype(bool)


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


def pixel_rays(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    pixels = np.atleast_2d(np.asarray(pixels, float))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.column_stack(
        ((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels)))
    )


def intersect_plane(rays: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    denominators = rays @ normal
    if np.any(np.abs(denominators) < 1e-9):
        raise ValueError("a keypoint ray is parallel to the door plane")
    t = d / denominators
    if np.any(t <= 0):
        raise ValueError("a keypoint ray hits the door plane behind the camera")
    return rays * t[:, None]


@dataclass(frozen=True)
class Door:
    type: DoorType
    handle_type: HandleType
    handle: np.ndarray
    corners: dict
    normal: np.ndarray  # unit, toward the camera
    handle_from_depth: bool = False  # False: the handle is on the door plane

    def transform(self, rotation, translation) -> "Door":
        rotation, translation = np.asarray(rotation), np.asarray(translation)
        return replace(
            self,
            handle=rotation @ self.handle + translation,
            corners={n: rotation @ p + translation for n, p in self.corners.items()},
            normal=rotation @ self.normal,
        )


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
