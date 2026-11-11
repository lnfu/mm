import json
import re
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .door import CORNERS, DoorType, HandleType

SIDES = ("left", "right")


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

    def keypoints(self, image, prompt, size, cancel) -> DoorKeypoints:
        text = self._client.ask(
            encode_jpeg(image, self._max_side, self._jpeg_quality),
            "image/jpeg",
            build_prompt(prompt),
            cancel,
        )
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
