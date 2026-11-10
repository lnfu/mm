from dataclasses import dataclass


@dataclass(frozen=True)
class GripperSettings:
    joint: str = "gripper_joint"  # FollowJointTrajectory command
    state_joint: str = "gripper_finger_left_joint"  # /joint_states
    open: float = 80.0  # driver units, past the mechanical stop
    closed: float = -10.0  # driver units, past the mechanical stop
    closed_finger: float = 0.0  # rad, finger reading when closed on nothing
    holding_margin: float = 0.05  # rad
    still: float = 0.005  # rad
    stall_time: float = 0.3  # s
    timeout: float = 5.0  # s
