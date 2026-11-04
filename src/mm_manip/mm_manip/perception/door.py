"""Door keypoints (from Gemini Robotics-ER), and their 3D position on the door plane.

No ROS, no network (the Gemini client is passed in).

A KeypointSource points to the handle and the four corners of the door in the
color image; GeminiDoorKeypoints is the one in use, a local model can replace it.
Corners sit on depth edges where single pixels often read the background or 0, so
the 3D points are not read from depth directly: a plane is fitted to the depth
inside the door, and each keypoint's camera ray is intersected with that plane.
"""

import json
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import cv2
import numpy as np

from .localize import mask_points

CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


class DoorType(IntEnum):
    """Same values as the TYPE_* in mm_interfaces/action/Articulate result."""

    UNKNOWN = 0
    REVOLUTE_LEFT = 1  # hinge on the left, as seen by the robot
    REVOLUTE_RIGHT = 2
    PRISMATIC = 3  # drawer


class HandleType(IntEnum):
    """Same values as the HANDLE_* in mm_interfaces/action/Articulate result."""

    UNKNOWN = 0
    BAR_VERTICAL = 1
    BAR_HORIZONTAL = 2
    KNOB = 3  # a single protruding knob
    RECESSED = 4  # recessed or none (push to open)


_TYPE_NAMES = {
    "revolute_left": DoorType.REVOLUTE_LEFT,
    "revolute_right": DoorType.REVOLUTE_RIGHT,
    "prismatic": DoorType.PRISMATIC,
    "unknown": DoorType.UNKNOWN,
}


_HANDLE_NAMES = {h.name.lower(): h for h in HandleType}


@dataclass(frozen=True)
class DoorKeypoints:
    """Pixel (u, v) coordinates in the image Gemini was given."""

    type: DoorType
    handle_type: HandleType
    handle: tuple
    corners: dict  # CORNERS name -> (u, v)


class KeypointSource(Protocol):
    def keypoints(
        self, image: np.ndarray, prompt: str, size: tuple, should_cancel
    ) -> DoorKeypoints:
        """The door `prompt` describes in `image` (BGR), as pixels of an image of
        `size` (width, height), e.g. the depth image's, which may differ from it.
        Raises ValueError (nothing usable), Canceled, or its own errors."""
        ...


class GeminiDoorKeypoints:
    """KeypointSource asking Gemini Robotics-ER (via a GeminiClient)."""

    def __init__(self, client, max_side: int = 640, jpeg_quality: int = 90):
        """
        Args:
            client: has ask(image bytes, mime_type, prompt, should_cancel) -> text.
            max_side: the image is shrunk to this longest side before upload
                (faster; Gemini answers in 0-1000 coordinates whatever the size).
        """
        self._client = client
        self._max_side = max_side
        self._jpeg_quality = jpeg_quality

    def keypoints(self, image, prompt, size, should_cancel) -> DoorKeypoints:
        text = self._client.ask(
            encode_jpeg(image, self._max_side, self._jpeg_quality),
            "image/jpeg",
            build_prompt(prompt),
            should_cancel,
        )
        return parse_response(text, *size)


def encode_jpeg(image: np.ndarray, max_side: int, quality: int = 90) -> bytes:
    """JPEG of `image`, shrunk (never enlarged) to `max_side` pixels on its longest side."""
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
  "bottom_left": [y, x], "bottom_right": [y, x]}}
type: revolute_left if the door's hinge is on its left side in the image,
revolute_right if on its right side, prismatic for a drawer.
handle_type: bar_vertical or bar_horizontal for a bar handle (by its long axis),
knob for a single protruding knob, recessed for a recessed handle or none.
handle: the center of its handle. top_left ... bottom_right: the corners of the
door's front panel (the moving part), as they appear in the image.
Points are [y, x], normalized to 0-1000."""


def parse_response(text: str, width: int, height: int) -> DoorKeypoints:
    """Gemini's answer to build_prompt() -> pixel keypoints in a width x height image.

    Raises ValueError if the answer is not the JSON asked for.
    """
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
        y, x = point
        return (x / 1000.0 * width, y / 1000.0 * height)

    return DoorKeypoints(
        door_type,
        handle_type,
        pixel("handle"),
        {name: pixel(name) for name in CORNERS},
    )


def quad_mask(corners: dict, shape, shrink: float = 0.15) -> np.ndarray:
    """Pixels inside the corner quadrilateral, shrunk toward its center by `shrink`
    (a fraction) to stay off the depth edges."""
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
    """Plane through the bulk of (N, 3) points: (unit normal n, d) with n.p = d.

    RANSAC with MSAC scoring: candidate planes through 3 random points are scored
    by the sum of min(distance^2, inlier_distance^2), so a plane wins by fitting
    its inliers tightly, not only by having many. It stops once `confidence` says
    a better plane is unlikely (about 17 tries at 70 % inliers). The best plane's
    inliers are then fitted by least squares. Points that are not the door (a
    handle sticking out, a neighbouring panel inside a loose quad) are ignored as
    long as the door is the largest plane.
    The normal faces the camera at the origin, so d < 0.
    Raises ValueError with fewer than 3 points or no plane.
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
        if norm < 1e-9:  # collinear
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
    if d > 0:  # make the normal face the camera, at the origin
        normal, d = -normal, -d
    return normal, d


def ransac_iterations(inlier_ratio: float, confidence: float, cap: int) -> int:
    """Tries needed to draw 3 inliers at least once with probability `confidence`."""
    p_good = inlier_ratio**3
    if p_good >= 1.0:
        return 1
    if p_good <= 0.0:
        return cap
    return min(cap, int(np.ceil(np.log(1.0 - confidence) / np.log1p(-p_good))))


def _least_squares_plane(points: np.ndarray):
    centroid = points.mean(axis=0)
    # full_matrices=False: U would otherwise be N x N (tens of GB for a door).
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][-1]
    return normal, float(normal @ centroid)


def pixel_rays(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 2) pixels (u, v) -> (N, 3) ray directions in the camera optical frame."""
    pixels = np.atleast_2d(np.asarray(pixels, float))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.column_stack(
        ((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels)))
    )


def intersect_plane(rays: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Where rays from the camera origin hit the plane n.p = d, as (N, 3) points.

    Raises ValueError for rays parallel to the plane or hitting it behind the camera.
    """
    denominators = rays @ normal
    if np.any(np.abs(denominators) < 1e-9):
        raise ValueError("a keypoint ray is parallel to the door plane")
    t = d / denominators
    if np.any(t <= 0):
        raise ValueError("a keypoint ray hits the door plane behind the camera")
    return rays * t[:, None]


@dataclass(frozen=True)
class Door:
    """A door in one frame (the camera's, or the base's after transform())."""

    type: DoorType
    handle_type: HandleType
    handle: np.ndarray  # (3,)
    corners: dict  # CORNERS name -> (3,)
    normal: np.ndarray  # (3,) unit, out of the door toward the camera

    def transform(self, rotation, translation) -> "Door":
        rotation, translation = np.asarray(rotation), np.asarray(translation)
        return Door(
            self.type,
            self.handle_type,
            rotation @ self.handle + translation,
            {n: rotation @ p + translation for n, p in self.corners.items()},
            rotation @ self.normal,
        )


def locate_door(
    keypoints: DoorKeypoints,
    depth: np.ndarray,
    K: np.ndarray,
    depth_band: float = 0.15,
    shrink: float = 0.15,
) -> Door:
    """Keypoints + depth -> Door in the camera optical frame.

    The keypoints are pixels of `depth` (parse_response with the depth image's
    size: Gemini's 0-1000 coordinates do not depend on the resolution), and K the
    intrinsics at that resolution. Depth registered to the color camera.
    Raises ValueError (too little depth in the door, degenerate geometry).
    """
    mask = quad_mask(keypoints.corners, depth.shape, shrink=shrink)
    points = mask_points(mask, depth, K, depth_band=depth_band)
    normal, d = fit_plane(points)
    pixels = np.array([keypoints.handle] + [keypoints.corners[n] for n in CORNERS])
    on_plane = intersect_plane(pixel_rays(pixels, K), normal, d)
    return Door(
        keypoints.type,
        keypoints.handle_type,
        on_plane[0],
        dict(zip(CORNERS, on_plane[1:], strict=True)),
        normal,
    )
