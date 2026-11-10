"""ROS parameters as frozen dataclasses. See NOTE.md."""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import ClassVar

from mm_ik.joints import ARM_JOINT_NAMES

from .motion.gripper import GripperSettings
from .motion.trajectory import DEFAULT_MAX_STEP


class Parameters:
    """declare() and from_node() for a frozen dataclass, under PREFIX.

    A dataclass field is a group of parameters `<field>.<its field>`. A dict field
    is one parameter per key, `<field>.<key>`.
    """

    PREFIX: ClassVar[str] = ""

    @classmethod
    def declare(cls, node) -> None:
        for name, value in _flatten(cls(), cls.PREFIX):
            node.declare_parameter(name, value)

    @classmethod
    def from_node(cls, node):
        return _read(cls(), cls.PREFIX, node)


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _flatten(value, prefix: str):
    if is_dataclass(value):
        for f in fields(value):
            yield from _flatten(getattr(value, f.name), _join(prefix, f.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield _join(prefix, key), item
    else:
        yield prefix, value


def _read(default, prefix: str, node):
    values = {}
    for f in fields(default):
        name, current = _join(prefix, f.name), getattr(default, f.name)
        if is_dataclass(current):
            values[f.name] = _read(current, name, node)
        elif isinstance(current, dict):
            values[f.name] = {
                key: node.get_parameter(_join(name, key)).value for key in current
            }
        else:
            values[f.name] = node.get_parameter(name).value
    return type(default)(**values)


@dataclass(frozen=True)
class Frames:
    base: str = "base_link"
    ee: str = "grasp_center_link"


@dataclass(frozen=True)
class RobotSettings:
    fjt_action: str = "/follow_joint_trajectory"
    driver_joint: dict = field(default_factory=lambda: {n: n for n in ARM_JOINT_NAMES})
    fjt_mode: str = "navigation"
    control_rate: float = 30.0  # Hz
    path_settle_timeout: float = 5.0  # s
    max_step: dict = field(default_factory=lambda: dict(DEFAULT_MAX_STEP))
    no_split_action_timeout: float = 10.0  # s


@dataclass(frozen=True)
class CameraSettings:
    color_topic: str = "/cameras_gripper/right/image_raw/compressed"
    depth_topic: str = "/cameras_gripper/stereo/image_raw/compressedDepth"
    depth_compressed: bool = True
    camera_info_topic: str = "/cameras_gripper/right/camera_info"
    sync_slop: float = 0.02  # s
    timeout: float = 2.0  # s


@dataclass(frozen=True)
class DetectSettings:
    service: str = "/mm_perception/detect"
    timeout: float = 5.0  # s
    depth_band: float = 0.05  # m


@dataclass(frozen=True)
class StartupConfig(Parameters):
    """Read once at startup. Restart the server to change it."""

    frames: Frames = field(default_factory=Frames)
    robot: RobotSettings = field(default_factory=RobotSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    detect: DetectSettings = field(default_factory=DetectSettings)


@dataclass(frozen=True)
class GoalConfig(Parameters):
    """Read at the start of each goal."""

    dry_run: bool = True
    settle_time: float = 0.3  # s
    use_gripper: bool = True
    gripper: GripperSettings = field(default_factory=GripperSettings)


@dataclass(frozen=True)
class IKAccept:
    position_error: float = 0.015  # m
    orientation_error_deg: float = 3.0


@dataclass(frozen=True)
class BaseLimits:
    translation: float = 0.6  # m, per axis
    rotation_deg: float = 45.0
