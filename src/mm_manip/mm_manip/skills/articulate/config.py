from dataclasses import dataclass, field
from typing import ClassVar

from mm_ik.joints import ARM_JOINT_NAMES

from ...config import BaseLimits, IKAccept, Parameters
from ...motion.velocity import TrackingGains


@dataclass(frozen=True)
class GeminiSettings:
    model: str = "gemini-robotics-er-2-preview"
    thinking_level: str = "low"  # minimal | low | medium | high
    timeout: float = 20.0  # s
    image_max_side: int = 640  # px


@dataclass(frozen=True)
class DoorSettings:
    quad_shrink: float = 0.15
    depth_band: float = 0.15  # m
    handle_offset: float = 0.03  # m
    handle_grasp_depth: float = 0.015  # m
    debug_dir: str = ""  # "" = <ROS log dir>/door_debug


@dataclass(frozen=True)
class ArticulateConfig(Parameters):
    PREFIX: ClassVar[str] = "articulate"

    start_pose: dict = field(
        default_factory=lambda: dict.fromkeys(ARM_JOINT_NAMES, 0.0)
    )
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    door: DoorSettings = field(default_factory=DoorSettings)
    n_waypoints: int = 10
    base_turn_ratios: list = field(default_factory=lambda: [0.3, 0.5, 0.7])
    approach_arm: float = 0.25  # m
    ik: IKAccept = field(default_factory=lambda: IKAccept(position_error=0.01))
    base_limits: BaseLimits = field(
        default_factory=lambda: BaseLimits(rotation_deg=90.0)
    )
    path: TrackingGains = field(default_factory=TrackingGains)
