"""Pinocchio model used for IK.

The base is pinocchio's planar root joint (x, y, theta), placed in front of the
URDF root. A planar joint is all or nothing: `base_dofs` is either empty (base
locked) or all of ('x', 'y', 'theta').

The telescoping arm (arm_l1..arm_l4) is merged into one joint: arm_l1..arm_l3
are locked at 0 and arm_l4 gets the summed range. The four joints share the same
axis, so this matches the official `merge_arm` in stretch4_urdf.
"""

import numpy as np
import pinocchio as pin

# Planar base joint. Its configuration is (x, y, cos(theta), sin(theta)) and its
# tangent is (vx, vy, wz); it has no limits.
BASE_JOINT = "base_planar_joint"
# Pinocchio's name for the joint passed as `root_joint`.
_PIN_ROOT_JOINT = "root_joint"
BASE_DOFS = ("x", "y", "theta")
ARM_SEGMENTS = ["arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint"]
# The merged arm lives on the last segment in the reduced model.
MERGED_ARM = ARM_SEGMENTS[-1]
ARM_JOINTS = [
    "lift_joint",
    MERGED_ARM,
    "wrist_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
]


def build_model(urdf_xml: str, base_dofs=(), joint_limits=None) -> pin.Model:
    """Return the reduced model with only lift, merged arm, wrist and the base (if any) free.

    Args:
        urdf_xml: Robot URDF, e.g. the `/robot_description` string.
        base_dofs: () to lock the base, or ('x', 'y', 'theta') to free it.
        joint_limits: Optional {joint_name: (lower, upper)} to tighten limits
            after the arm merge (use MERGED_ARM for the arm). Other joints
            (e.g. the base) are ignored.
    """
    if sorted(base_dofs) not in ([], sorted(BASE_DOFS)):
        raise ValueError(
            f"base_dofs must be () or {BASE_DOFS} with a planar base, got {tuple(base_dofs)}"
        )

    full = pin.buildModelFromXML(urdf_xml, pin.JointModelPlanar())
    missing = [j for j in ARM_SEGMENTS + ARM_JOINTS if not full.existJointName(j)]
    if missing:
        raise ValueError(f"URDF is missing joints {missing}")

    arm_upper = sum(
        full.upperPositionLimit[full.idx_qs[full.getJointId(j)]] for j in ARM_SEGMENTS
    )

    keep = set(ARM_JOINTS) | ({_PIN_ROOT_JOINT} if base_dofs else set())
    lock_ids = [full.getJointId(n) for n in full.names[1:] if n not in keep]
    model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
    if base_dofs:
        # pink treats a joint named "root_joint" as a floating base and leaves it
        # out of PostureTask; rename it so the posture costs apply to the base too.
        model.names[model.getJointId(_PIN_ROOT_JOINT)] = BASE_JOINT

    model.upperPositionLimit[model.idx_qs[model.getJointId(MERGED_ARM)]] = arm_upper
    for name, (lower, upper) in (joint_limits or {}).items():
        if name not in ARM_JOINTS:
            continue
        idx = model.idx_qs[model.getJointId(name)]
        model.lowerPositionLimit[idx] = max(model.lowerPositionLimit[idx], lower)
        model.upperPositionLimit[idx] = min(model.upperPositionLimit[idx], upper)
    return model
