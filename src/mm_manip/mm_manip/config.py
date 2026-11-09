"""Server parameters.

ServerConfig is shared by every skill and named at the top level; each skill's own
parameters live under its prefix, e.g. `grasp.lift_height`. Dict fields are ROS
parameters named `<field>.<key>`, e.g. `grasp.start_pose.lift_joint`.
"""

from dataclasses import asdict, dataclass, field
from typing import ClassVar

from mm_ik.solver import ARM_JOINT_NAMES

from .motion.trajectory import DEFAULT_MAX_STEP
from .motion.velocity import DEFAULT_MAX_VELOCITY, DEFAULT_TOLERANCE

WRIST_JOINTS = ["wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]


class _Parameters:
    """declare()/from_node() for a frozen dataclass, under the class's PREFIX."""

    PREFIX: ClassVar[str] = ""

    @classmethod
    def _name(cls, name: str) -> str:
        return f"{cls.PREFIX}.{name}" if cls.PREFIX else name

    @classmethod
    def declare(cls, node) -> None:
        """Declare every parameter on `node`, with the defaults of the dataclass."""
        for name, default in asdict(cls()).items():
            if isinstance(default, dict):
                for key, value in default.items():
                    node.declare_parameter(cls._name(f"{name}.{key}"), value)
            else:
                node.declare_parameter(cls._name(name), default)

    @classmethod
    def from_node(cls, node):
        """The current values of the parameters declared by declare()."""
        values = {}
        for name, default in asdict(cls()).items():
            if isinstance(default, dict):
                values[name] = {
                    key: node.get_parameter(cls._name(f"{name}.{key}")).value
                    for key in default
                }
            else:
                values[name] = node.get_parameter(cls._name(name)).value
        return cls(**values)


@dataclass(frozen=True)
class ServerConfig(_Parameters):
    """Shared by every skill. Two kinds of parameters, see the section comments."""

    # ---- Read at the start of each goal: `ros2 param set` applies to the next goal.

    dry_run: bool = True  # plan + RViz only, nothing moves
    ee_frame: str = "grasp_center_link"
    settle_time: float = 0.3  # s to wait after the arm stops, before detecting
    # false: skip opening, closing and the "holding something" check, but still move.
    use_gripper: bool = True

    # ---- Read once at startup to build Robot, Scene and Locator: restart to change.

    base_frame: str = "base_link"
    fjt_action: str = "/follow_joint_trajectory"  # served by /stretch_driver
    # IK joint name -> name the driver uses (the same names on Stretch 4).
    driver_joint: dict = field(default_factory=lambda: {n: n for n in ARM_JOINT_NAMES})
    # If stretch_driver's action_timeout is at least this long, long moves are not
    # split into max_step pieces (fewer stops).
    no_split_action_timeout: float = 10.0
    max_step: dict = field(default_factory=lambda: dict(DEFAULT_MAX_STEP))
    # stretch_driver mode for FollowJointTrajectory moves. navigation also takes
    # /cmd_vel (grasp's base moves); position does not. Velocity paths switch the
    # driver to velocity mode and the next FJT move switches it back.
    fjt_mode: str = "navigation"
    control_rate: float = 30.0  # Hz, velocity path commands
    path_settle_timeout: float = 5.0  # s allowed after a velocity path ends to arrive
    # Gripper through FollowJointTrajectory, driver units past both mechanical stops.
    gripper_joint: str = "gripper_joint"
    gripper_state_joint: str = "gripper_finger_left_joint"  # read in /joint_states
    gripper_open: float = 80.0
    gripper_closed: float = -10.0
    # Finger reading (rad) when closed on nothing; measure it on the robot.
    gripper_closed_finger: float = 0.0  # TODO[lnfu] measure
    gripper_holding_margin: float = 0.05  # rad farther than this = holding something
    # Holding something: the fingers stop short of closed for this long.
    gripper_stall_time: float = 0.3  # s
    gripper_timeout: float = 5.0  # s

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


@dataclass(frozen=True)
class GraspConfig(_Parameters):
    """The grasp skill, read at the start of each goal."""

    PREFIX: ClassVar[str] = "grasp"

    base_dofs: str = ""  # "" | "x" | "x,y" | "x,y,theta"
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

    gripper_max_opening: float = 0.10  # TODO[lnfu] measure

    @property
    def base_dof_list(self) -> list:
        """base_dofs as a list, e.g. "x, y" -> ["x", "y"]."""
        return [d.strip() for d in self.base_dofs.split(",") if d.strip()]


@dataclass(frozen=True)
class ArticulateConfig(_Parameters):
    """The articulate skill (open a door), read at the start of each goal."""

    PREFIX: ClassVar[str] = "articulate"

    start_pose: dict = field(
        default_factory=lambda: dict.fromkeys(ARM_JOINT_NAMES, 0.0)
    )

    # Gemini Robotics-ER finds the handle and the door's corners (MODE_PROMPT).
    gemini_model: str = "gemini-robotics-er-2-preview"
    gemini_thinking_level: str = "low"  # minimal | low | medium | high: slower, better
    gemini_timeout: float = 20.0  # s
    image_max_side: int = 640  # px, the image is shrunk to this before upload
    # The door plane is fitted to the depth inside the corner quad, shrunk by
    # quad_shrink toward its center, within door_depth_band of the median depth.
    quad_shrink: float = 0.15
    door_depth_band: float = 0.15  # m
    # The handle keypoint lands on the door plane; the handle sticks out by this.
    handle_offset: float = 0.03  # m, MODE_PROMPT only

    n_waypoints: int = 10  # on the opening arc, the grasp pose included
    # The base (x, y, theta) moves during the arc where the arm alone cannot follow
    # it. The arc is planned with the base's turn left to the IK, and held to each
    # of these shares of the door's turn; the plan opening furthest wins.
    base_turn_ratios: list = field(default_factory=lambda: [0.3, 0.5, 0.7])
    max_base_translation: float = 0.6  # m, per axis
    max_base_rotation_deg: float = 90.0
    # When the handle is out of the arm's reach, the base moves first (with the arm,
    # in one velocity path) so the arm grasps near this extension (max 0.52).
    approach_arm: float = 0.25
    ik_accept_position_error: float = 0.01  # m
    ik_accept_orientation_error_deg: float = 3.0

    # Velocity paths (all articulate motions): m/s or rad/s per joint, base_x and
    # base_y included, and how close each joint must end (m or rad).
    path_max_velocity: dict = field(default_factory=lambda: dict(DEFAULT_MAX_VELOCITY))
    path_tolerance: dict = field(default_factory=lambda: dict(DEFAULT_TOLERANCE))
    tracking_gain: float = 2.0  # 1/s, 0 = feedforward only
    path_min_segment: float = 0.3  # s, shortest time between two waypoints
