"""Grasp server parameters

Dict fields are ROS parameters named `<field>.<joint>`.
"""

from dataclasses import asdict, dataclass, field

from mm_ik.solver import ARM_JOINT_NAMES

from .motion.trajectory import DEFAULT_MAX_STEP

WRIST_JOINTS = ["wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]


@dataclass(frozen=True)
class GraspConfig:
    """Two kinds of parameters, see the section comments below."""

    # ---- Read at the start of each goal: `ros2 param set` applies to the next goal.

    dry_run: bool = True  # plan + RViz only, nothing moves
    base_dofs: str = ""  # "" | "x" | "x,y" | "x,y,theta"
    ee_frame: str = "grasp_center_link"
    start_pose: dict = field(
        default_factory=lambda: dict.fromkeys(ARM_JOINT_NAMES, 0.0)
    )
    grasp_wrist: dict = field(default_factory=lambda: dict.fromkeys(WRIST_JOINTS, 0.0))

    pregrasp_distance: float = 0.10
    # stretch_driver stops at every point; arm-only approach is already straight.
    n_waypoints: int = 2
    lift_height: float = 0.10
    # Let wrist yaw aim at objects off the gripper's line (base locked), keeping the
    # gripper level. The approach is still straight along base x.
    free_yaw: bool = True
    # ~5 cm sideways; anything further is left to the base.
    max_aim_yaw_deg: float = 10.0
    # The base is only used when the arm alone cannot reach (base_dofs not empty).
    # Then it moves so the arm ends near base_stage_arm at the pre-grasp, leaving
    # room for the approach (0.52 max - pregrasp_distance - margin).
    base_stage_arm: float = 0.35
    max_base_translation: float = 0.6  # m, per axis
    max_base_rotation_deg: float = 45.0
    # Hello Robot's sentry caps the base at 0.1 m/s while the arm is out (extended
    # past 3 cm or lifted above 0.3 m), which is always the case while grasping.
    base_max_linear: float = 0.1  # m/s
    base_max_angular: float = 0.5  # rad/s
    # Speed = gain * remaining distance (or angle), so the gain sets how long the
    # base takes to creep over its last centimetres. Too high overshoots.
    base_linear_gain: float = 2.0  # 1/s
    base_angular_gain: float = 2.0  # 1/s
    # After each base move the object is detected again and the grasp replanned,
    # with the base allowed again: a first look near the image edge can be far off.
    max_base_moves: int = 3
    # A target slightly off the gripper's line (base locked) never converges exactly.
    ik_accept_position_error: float = 0.015  # m
    ik_accept_orientation_error_deg: float = 3.0

    settle_time: float = 0.3  # s to wait after the arm stops, before detecting

    # TODO[lnfu] the gripper mechanism is broken; open/close/check are skipped
    # while use_gripper is false and the arm motions still run.
    use_gripper: bool = False
    gripper_max_opening: float = 0.10  # TODO[lnfu] measure
    closed_threshold: float = 0.0  # TODO[lnfu] from the empty vs holding test

    # ---- Read once at startup to build Robot, Scene and Locator: restart to change.

    base_frame: str = "base_link"
    fjt_action: str = "/follow_joint_trajectory"  # served by /stretch_driver
    # IK joint name -> name the driver uses (the same names on Stretch 4).
    driver_joint: dict = field(default_factory=lambda: {n: n for n in ARM_JOINT_NAMES})
    # If stretch_driver's action_timeout is at least this long, long moves are not
    # split into max_step pieces (fewer stops).
    no_split_action_timeout: float = 10.0
    max_step: dict = field(default_factory=lambda: dict(DEFAULT_MAX_STEP))

    detect_service: str = "/mm_perception/detect"
    service_timeout: float = 5.0
    # The driver aligns depth to the right camera (setDepthAlign(right)) and
    # publishes it only as compressedDepth, so the color image is the right one.
    color_topic: str = "/cameras_gripper/right/image_raw/compressed"
    depth_topic: str = "/cameras_gripper/stereo/image_raw/compressedDepth"
    depth_compressed: bool = True
    camera_info_topic: str = "/cameras_gripper/right/camera_info"
    sync_slop: float = 0.02  # s, max stamp difference within a color/depth/info triple
    snapshot_timeout: float = 2.0
    depth_band: float = 0.05  # m, drop mask pixels this far from the median depth

    @property
    def base_dof_list(self) -> list:
        """base_dofs as a list, e.g. "x, y" -> ["x", "y"]."""
        return [d.strip() for d in self.base_dofs.split(",") if d.strip()]

    @classmethod
    def declare(cls, node) -> None:
        """Declare every parameter on `node`, with the defaults above."""
        for name, default in asdict(cls()).items():
            if isinstance(default, dict):
                for key, value in default.items():
                    node.declare_parameter(f"{name}.{key}", value)
            else:
                node.declare_parameter(name, default)

    @classmethod
    def from_node(cls, node) -> "GraspConfig":
        """The current values of the parameters declared by declare()."""
        values = {}
        for name, default in asdict(cls()).items():
            if isinstance(default, dict):
                values[name] = {
                    key: node.get_parameter(f"{name}.{key}").value for key in default
                }
            else:
                values[name] = node.get_parameter(name).value
        return cls(**values)
