"""Joint names used outside mm_ik. No pinocchio."""

ARM_JOINT_NAMES = [
    "lift_joint",
    "arm_joint",
    "wrist_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
]
# Base travel from the start pose, in the start base frame (m, m, rad).
BASE_X = "base_x"
BASE_Y = "base_y"
BASE_THETA = "base_theta"
BASE_JOINTS = (BASE_X, BASE_Y, BASE_THETA)
# The telescoping arm in the URDF and /joint_states; arm_joint is their sum.
ARM_SEGMENTS = ["arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint"]


def arm_only(joints: dict) -> dict:
    return {k: joints[k] for k in ARM_JOINT_NAMES}


def base_only(joints: dict) -> dict:
    return {k: v for k, v in joints.items() if k in BASE_JOINTS}
