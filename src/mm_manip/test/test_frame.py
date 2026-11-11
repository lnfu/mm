import numpy as np
import pinocchio as pin

from mm_manip.camera.frame import Frame

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 200.0], [0.0, 0.0, 1.0]])
# Camera 1 m above the base origin, looking along base +x (optical z = base x).
CAMERA_TO_BASE = pin.SE3(
    np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    np.array([0.0, 0.0, 1.0]),
)


def frame():
    return Frame(
        color=np.zeros((400, 640, 3), np.uint8),
        color_data=b"",
        color_format="jpeg",
        depth=np.zeros((400, 640), np.uint16),
        K=K,
        camera_to_base=CAMERA_TO_BASE,
        frame_id="camera",
        stamp_ns=0,
        age=0.0,
    )


def test_depth_size_is_width_height():
    assert frame().depth_size == (640, 400)
