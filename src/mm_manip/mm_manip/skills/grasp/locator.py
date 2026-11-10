from sensor_msgs.msg import CompressedImage

from mm_interfaces.srv import Detect

from ...camera.camera import Camera, header
from ...camera.images import decode_mask_png
from ...cancel import CancelToken
from ...errors import PerceptionError
from ...geometry.points import mask_points, touches_edge, transform_points
from .skill import Observation


class Locator:
    def __init__(
        self,
        node,
        camera: Camera,
        detect_service,
        service_timeout=5.0,
        depth_band=0.05,
        callback_group=None,
    ):
        self._node = node
        self._camera = camera
        self._service_timeout = service_timeout
        self._depth_band = depth_band
        self._detect = node.create_client(
            Detect, detect_service, callback_group=callback_group
        )

    def locate(self, prompt: str, cancel: CancelToken) -> Observation:
        frame = self._camera.capture(cancel)
        image = CompressedImage(
            header=header(frame), format=frame.color_format, data=frame.color_data
        )
        res = self._call(Detect.Request(prompt=prompt, image=image), cancel)
        if not res.success:
            raise PerceptionError(f"detect failed: {res.message}")
        if not res.detections:
            raise PerceptionError(f'no "{prompt}" found')
        best = res.detections[0]  # sorted by score
        self._node.get_logger().info(f"detected {best.label} ({best.score:.2f})")

        mask = decode_mask_png(best.mask.data)
        truncated = touches_edge(mask)
        if truncated:
            self._node.get_logger().warn(
                f"{best.label} touches the image edge: it may be cut off, "
                "so its position is uncertain"
            )
        try:
            points = mask_points(
                mask, frame.depth, frame.K, depth_band=self._depth_band
            )
        except ValueError as e:
            raise PerceptionError(f"{best.label}: {e}") from e
        points = transform_points(points, frame.camera_to_base)
        return Observation(best.label, best.score, points, truncated)

    def _call(self, request, cancel: CancelToken):
        if not self._detect.wait_for_service(timeout_sec=self._service_timeout):
            raise PerceptionError(f"service {self._detect.srv_name} not available")
        future = self._detect.call_async(request)
        if not cancel.wait_until(future.done, self._service_timeout):
            raise PerceptionError(f"service {self._detect.srv_name} timed out")
        return future.result()
