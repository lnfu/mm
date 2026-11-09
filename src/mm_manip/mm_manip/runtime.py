from dataclasses import dataclass

from rclpy.callback_groups import ReentrantCallbackGroup

from .camera.camera import Camera
from .config import ServerConfig
from .motion.robot import GripperSettings, Robot
from .scene import Scene


@dataclass(frozen=True)
class Runtime:
    robot: Robot
    scene: Scene
    camera: Camera
    callback_group: ReentrantCallbackGroup


def build(node, cfg: ServerConfig) -> Runtime:
    group = ReentrantCallbackGroup()
    scene = Scene(node, base_frame=cfg.base_frame)
    robot = Robot(
        node,
        cfg.fjt_action,
        cfg.driver_joint,
        max_step=cfg.max_step,
        no_split_action_timeout=cfg.no_split_action_timeout,
        fjt_mode=cfg.fjt_mode,
        control_rate=cfg.control_rate,
        path_settle_timeout=cfg.path_settle_timeout,
        gripper=gripper_settings(cfg),
        callback_group=group,
    )
    camera = Camera(
        node,
        scene,
        cfg.color_topic,
        cfg.depth_topic,
        cfg.camera_info_topic,
        depth_compressed=cfg.depth_compressed,
        sync_slop=cfg.sync_slop,
        timeout=cfg.snapshot_timeout,
        callback_group=group,
    )
    return Runtime(robot, scene, camera, group)


def gripper_settings(cfg: ServerConfig) -> GripperSettings:
    return GripperSettings(
        joint=cfg.gripper_joint,
        state_joint=cfg.gripper_state_joint,
        open=cfg.gripper_open,
        closed=cfg.gripper_closed,
        closed_finger=cfg.gripper_closed_finger,
        holding_margin=cfg.gripper_holding_margin,
        stall_time=cfg.gripper_stall_time,
        timeout=cfg.gripper_timeout,
    )
