import numpy as np
import pinocchio as pin


def transform_to_se3(transform) -> pin.SE3:
    """geometry_msgs/Transform -> pin.SE3."""
    q, t = transform.rotation, transform.translation
    rotation = pin.Quaternion(q.w, q.x, q.y, q.z).normalized().toRotationMatrix()
    return pin.SE3(rotation, np.array([t.x, t.y, t.z]))
