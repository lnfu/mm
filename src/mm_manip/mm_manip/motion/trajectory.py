import numpy as np

from mm_ik.joints import ARM_SEGMENTS

# Largest change per point: stretch_driver must finish it in action_timeout (3 s).
DEFAULT_MAX_STEP = {
    "lift_joint": 0.2,
    "arm_joint": 0.15,
    "wrist_yaw_joint": 0.8,
    "wrist_pitch_joint": 0.8,
    "wrist_roll_joint": 0.8,
}
# Smaller changes are not sent, see moving_joints.
MIN_JOINT_CHANGE = {
    "lift_joint": 0.002,
    "arm_joint": 0.002,
    "wrist_yaw_joint": 0.005,
    "wrist_pitch_joint": 0.005,
    "wrist_roll_joint": 0.005,
}


def urdf_joint_positions(joints: dict):
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
    # stretch_driver never completes a lift or arm target at the current position.
    out = []
    prev = dict(start)
    for p in points:
        moved = {j: v for j, v in p.items() if abs(v - prev[j]) > min_change[j]}
        if moved:
            out.append(moved)
            prev.update(moved)
    return out
