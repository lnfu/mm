import time

from ...camera.camera import Camera
from ...cancel import CancelToken
from ...errors import PerceptionError
from .door import Door
from .gemini import GeminiError
from .keypoints import KeypointSource, hinge_edge_problem
from .locate import locate_door


class DoorLocator:
    def __init__(self, node, camera: Camera):
        self._node = node
        self._camera = camera

    def locate(
        self,
        prompt: str,
        source: KeypointSource,
        cancel: CancelToken,
        depth_band=0.15,
        shrink=0.15,
    ) -> Door:
        """The door `prompt` describes, in the base frame. The camera must be still."""
        t0 = time.monotonic()
        frame = self._camera.capture(cancel)
        t1 = time.monotonic()

        try:
            keypoints = source.keypoints(frame.color, prompt, frame.depth_size, cancel)
        except (GeminiError, ValueError) as e:
            raise PerceptionError(f"keypoints: {e}") from e
        t2 = time.monotonic()
        problem = hinge_edge_problem(keypoints)
        if problem is not None:
            raise PerceptionError(problem)

        try:
            door = locate_door(
                keypoints, frame.depth, frame.K, depth_band=depth_band, shrink=shrink
            )
        except ValueError as e:
            raise PerceptionError(f"door: {e}") from e
        door = door.transform(frame.camera_to_base)
        t3 = time.monotonic()

        self._node.get_logger().info(
            f"door {door.type.name}, handle from "
            f"{'depth' if door.handle_from_depth else 'the door plane'}: "
            f"snapshot {t1 - t0:.3f} s (image {frame.age:.2f} s old), "
            f"keypoints {t2 - t1:.3f} s, plane {t3 - t2:.3f} s"
        )
        return door
