from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
import pinocchio as pin

from ...errors import ArticulateFailed

CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


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


@dataclass(frozen=True)
class Door:
    type: DoorType
    handle_type: HandleType
    handle: np.ndarray
    corners: dict
    normal: np.ndarray  # unit, toward the camera
    handle_from_depth: bool = False  # False: the handle is on the door plane

    def transform(self, pose: pin.SE3) -> "Door":
        rotation, translation = pose.rotation, pose.translation
        return replace(
            self,
            handle=rotation @ self.handle + translation,
            corners={n: rotation @ p + translation for n, p in self.corners.items()},
            normal=rotation @ self.normal,
        )


Z = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Arc:
    axis: np.ndarray
    poses: list  # poses[0] grasps the handle
    angles: np.ndarray  # rad
    sign: float  # +1: the door turns counterclockwise, seen from above


def door_from_points(handle, hinge) -> Door:
    handle, hinge = np.asarray(handle, float), np.asarray(hinge, float)
    along = horizontal(hinge - handle)
    normal = np.cross(Z, along)
    if normal @ -horizontal(handle) < 0:
        normal = -normal
    door_type = (
        DoorType.REVOLUTE_LEFT if hinge[1] > handle[1] else DoorType.REVOLUTE_RIGHT
    )
    hinge_at_handle = np.array([hinge[0], hinge[1], handle[2]])
    corners = {"hinge": hinge_at_handle}
    return Door(door_type, HandleType.BAR_VERTICAL, handle, corners, normal)


def into_handle(door: Door, depth: float, clearance: float = 0.005):
    """Move the handle `depth` into the door. Returns (door, protrusion, moved) in m."""
    out = horizontal(door.normal)
    protrusion = float((door.handle - door.corners["top_left"]) @ door.normal)
    moved = min(depth, max(protrusion - clearance, 0.0))
    return replace(door, handle=door.handle - moved * out), protrusion, moved


def with_handle_offset(door: Door, offset: float) -> Door:
    return replace(door, handle=door.handle + offset * horizontal(door.normal))


def hinge_axis(door: Door) -> np.ndarray:
    if "hinge" in door.corners:
        point = door.corners["hinge"]
    else:
        side = "left" if door.type == DoorType.REVOLUTE_LEFT else "right"
        top, bottom = door.corners[f"top_{side}"], door.corners[f"bottom_{side}"]
        edge = bottom - top
        length = np.linalg.norm(edge)
        if length < 1e-6:
            raise ArticulateFailed(f"the {side} door corners coincide")
        edge /= length
        point = top + ((door.handle - top) @ edge) * edge
    return np.array([point[0], point[1], door.handle[2]])


def radius(door: Door, arc: Arc) -> float:
    return float(np.linalg.norm((door.handle - arc.axis)[:2]))


def opening_arc(door: Door, angle: float, n_waypoints: int, roll: float = 0.0) -> Arc:
    if n_waypoints < 2:
        raise ValueError("n_waypoints must be >= 2")
    axis = hinge_axis(door)
    arm = door.handle - axis
    if np.linalg.norm(arm[:2]) < 0.02:
        raise ArticulateFailed("the handle is on the hinge axis")
    # Turn so that the handle first moves toward the robot.
    sign = 1.0 if np.cross(Z, arm) @ door.normal > 0 else -1.0
    pointing = -horizontal(door.normal)
    start = _rot_z(np.arctan2(pointing[1], pointing[0])) @ _rot_x(roll)
    angles = np.linspace(0.0, angle, n_waypoints)
    poses = []
    for theta in angles:
        turn = _rot_z(sign * theta)
        poses.append(pin.SE3(turn @ start, axis + turn @ arm))
    return Arc(axis, poses, angles, sign)


def horizontal(v) -> np.ndarray:
    v = np.array([v[0], v[1], 0.0])
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ArticulateFailed("a door direction is vertical")
    return v / norm


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
