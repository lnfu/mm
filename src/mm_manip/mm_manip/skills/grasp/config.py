from dataclasses import dataclass, field
from typing import ClassVar

from mm_ik.joints import ARM_JOINT_NAMES

from ...config import BaseLimits, IKAccept, Parameters
from ...motion.base_control import BaseGains

WRIST_JOINTS = ["wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]


@dataclass(frozen=True)
class GraspConfig(Parameters):
    PREFIX: ClassVar[str] = "grasp"

    base_dofs: str = ""  # "" | "x" | "x,y" | "x,y,theta"
    start_pose: dict = field(
        default_factory=lambda: dict.fromkeys(ARM_JOINT_NAMES, 0.0)
    )
    grasp_wrist: dict = field(default_factory=lambda: dict.fromkeys(WRIST_JOINTS, 0.0))
    pregrasp_distance: float = 0.10  # m
    n_waypoints: int = 2
    lift_height: float = 0.10  # m
    free_yaw: bool = True
    max_aim_yaw_deg: float = 10.0
    base_stage_arm: float = 0.35  # m
    max_base_moves: int = 3
    gripper_max_opening: float = 0.10  # m, TODO[lnfu] measure
    ik: IKAccept = field(default_factory=IKAccept)
    base_limits: BaseLimits = field(default_factory=BaseLimits)
    base_control: BaseGains = field(
        default_factory=lambda: BaseGains(
            max_linear=0.1, max_angular=0.5, linear_gain=2.0, angular_gain=2.0
        )
    )

    @property
    def base_dof_list(self) -> list:
        return [d.strip() for d in self.base_dofs.split(",") if d.strip()]
