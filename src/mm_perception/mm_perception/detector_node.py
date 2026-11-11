"""YOLOE detector.

Services (under the node name, default /mm_perception/):
  ~/detect (mm_interfaces/Detect): detect `prompt` in the request image, return masks.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from mm_interfaces.msg import Detection
from mm_interfaces.srv import Detect

from .detections import encode_mask_png, parse_prompt, to_detections


class DetectorNode(Node):
    def __init__(self):
        super().__init__("mm_perception")
        self.declare_parameter("model", "yoloe-26s-seg.pt")
        self.declare_parameter(
            "device", ""
        )  # '' lets ultralytics pick (cuda:0 if available)
        self.declare_parameter("min_score", 0.3)

        # Load everything a first request would otherwise wait for: the text encoder
        # (downloaded on first use), its embeddings, and the CUDA kernels.
        self.declare_parameter("warmup_prompt", "cup")

        from ultralytics import YOLOE  # slow import, keep it out of module load

        self._model = YOLOE(self.get_parameter("model").value)
        self._classes = None
        warmup = parse_prompt(self.get_parameter("warmup_prompt").value)
        if warmup:
            self.get_logger().info(
                f"warming up with {warmup} (first run downloads the text encoder)"
            )
            self._model.set_classes(warmup)
            self._classes = warmup
            self._model.predict(
                np.zeros((400, 640, 3), np.uint8),
                verbose=False,
                device=self.get_parameter("device").value or None,
            )

        self.create_service(Detect, "~/detect", self._on_detect)
        self.get_logger().info("ready")

    def _on_detect(self, req, res):
        classes = parse_prompt(req.prompt)
        if not classes:
            return self._fail(res, "empty prompt")
        image = cv2.imdecode(
            np.frombuffer(bytes(req.image.data), np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            return self._fail(res, f'cannot decode image (format "{req.image.format}")')

        if classes != self._classes:
            self._model.set_classes(classes)  # computes text embeddings
            self._classes = classes
        device = self.get_parameter("device").value or None
        min_score = self.get_parameter("min_score").value
        result = self._model.predict(
            image, conf=min_score, retina_masks=True, verbose=False, device=device
        )[0]

        if result.masks is None or len(result.boxes) == 0:
            dets = []
        else:
            dets = to_detections(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.masks.data.cpu().numpy(),
                result.names,
                min_score,
            )

        for d in dets:
            msg = Detection(label=d.label, score=d.score, bbox=list(d.bbox))
            msg.mask.header = req.image.header
            msg.mask.format = "png"
            msg.mask.data = encode_mask_png(d.mask)
            res.detections.append(msg)
        res.success = True
        res.message = f"{len(dets)} detection(s) for {classes}"
        self.get_logger().info(res.message)
        return res

    def _fail(self, res, message):
        res.success = False
        res.message = message
        self.get_logger().warn(f"detect failed: {message}")
        return res


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
