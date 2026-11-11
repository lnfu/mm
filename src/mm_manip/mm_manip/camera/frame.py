from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class Frame:
    color: np.ndarray  # BGR
    color_data: bytes  # compressed, as published
    color_format: str
    depth: np.ndarray  # uint16 mm or float32 m, aligned to color
    K: np.ndarray  # (3, 3), for depth pixels
    camera_to_base: pin.SE3
    frame_id: str
    stamp_ns: int
    age: float  # s, from the image stamp to the capture

    @property
    def depth_size(self) -> tuple:
        return self.depth.shape[1], self.depth.shape[0]
