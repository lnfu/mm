"""Joint trajectories for stretch_driver: its limits, URDF joint names, the FJT message."""

import numpy as np

from mm_ik.model import ARM_SEGMENTS

# Largest change per trajectory point, so stretch_driver finishes each within its 3 s action_timeout.
DEFAULT_MAX_STEP = {
    "lift_joint": 0.2,
    "arm_joint": 0.15,
    "wrist_yaw_joint": 0.8,
    "wrist_pitch_joint": 0.8,
    "wrist_roll_joint": 0.8,
}
# Smaller changes are left out of a trajectory point (see moving_joints).
MIN_JOINT_CHANGE = {
    "lift_joint": 0.002,
    "arm_joint": 0.002,
    "wrist_yaw_joint": 0.005,
    "wrist_pitch_joint": 0.005,
    "wrist_roll_joint": 0.005,
}


def urdf_joint_positions(joints: dict):
    """IK joint dict -> (names, positions) with URDF names, e.g. for a ghost JointState.

    arm_joint is spread evenly over arm_l1..arm_l4, the way stretch_driver publishes it.
    """
    names, positions = [], []
    for name, value in joints.items():
        if name == "arm_joint":
            names += ARM_SEGMENTS
            positions += [value / len(ARM_SEGMENTS)] * len(ARM_SEGMENTS)
        else:
            names.append(name)
            positions.append(float(value))
    return names, positions


def joint_trajectory(names: list, points: list, times: list):
    """Build a trajectory_msgs/JointTrajectory. `points` are dicts keyed by `names`."""
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    traj = JointTrajectory(joint_names=list(names))
    for p, t in zip(points, times):
        sec = int(t)
        traj.points.append(
            JointTrajectoryPoint(
                positions=[float(p[n]) for n in names],
                time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9)),
            )
        )
    return traj


def split_long_moves(start: dict, points: list, max_step: dict) -> list:
    """Insert evenly spaced points so no joint moves more than max_step[j] per point.

    stretch_driver stops at every point and aborts a point it cannot finish within
    3 s (its action_timeout), so long moves are cut into pieces it can finish.
    The original points are kept exactly.
    """
    out = []
    prev = start
    for p in points:
        n = max(
            1, int(np.ceil(max(abs(p[j] - prev[j]) / max_step[j] for j in p) - 1e-9))
        )
        out += [{j: prev[j] + (p[j] - prev[j]) * k / n for j in p} for k in range(1, n)]
        out.append(dict(p))
        prev = p
    return out


def moving_joints(start: dict, points: list, min_change: dict) -> list:
    """Per point, only the joints that move more than min_change[j] since the last point.

    stretch_driver counts lift/arm as done only after their motion generator has moved,
    so a point that leaves one of them where it is never finishes (Goal timed out).
    Left-out joints simply stay put; points where nothing moves are dropped.
    """
    out = []
    prev = dict(start)
    for p in points:
        moved = {j: v for j, v in p.items() if abs(v - prev[j]) > min_change[j]}
        if moved:
            out.append(moved)
            prev.update(moved)
    return out
