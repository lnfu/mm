from dataclasses import dataclass

from rclpy.callback_groups import ReentrantCallbackGroup

from .camera.camera import Camera
from .config import StartupConfig
from .robot import Robot
from .scene import Scene


@dataclass(frozen=True)
class Runtime:
    robot: Robot
    scene: Scene
    camera: Camera
    callback_group: ReentrantCallbackGroup


def build(node, startup: StartupConfig) -> Runtime:
    group = ReentrantCallbackGroup()
    scene = Scene(node, base_frame=startup.frames.base)
    robot_cfg, camera_cfg = startup.robot, startup.camera
    robot = Robot(
        node,
        robot_cfg.fjt_action,
        robot_cfg.driver_joint,
        max_step=robot_cfg.max_step,
        no_split_action_timeout=robot_cfg.no_split_action_timeout,
        fjt_mode=robot_cfg.fjt_mode,
        control_rate=robot_cfg.control_rate,
        path_settle_timeout=robot_cfg.path_settle_timeout,
        callback_group=group,
    )
    camera = Camera(
        node,
        scene,
        camera_cfg.color_topic,
        camera_cfg.depth_topic,
        camera_cfg.camera_info_topic,
        depth_compressed=camera_cfg.depth_compressed,
        sync_slop=camera_cfg.sync_slop,
        timeout=camera_cfg.timeout,
        callback_group=group,
    )
    return Runtime(robot, scene, camera, group)
