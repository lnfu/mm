"""Pinocchio model used for IK.

The model always contains a virtual planar base (x, y, theta) in front of the
URDF root. `base_dofs` chooses which of those are free; the rest are locked at 0.

The telescoping arm (arm_l1..arm_l4) is merged into one joint: arm_l1..arm_l3
are locked at 0 and arm_l4 gets the summed range. The four joints share the same
axis, so this matches the official `merge_arm` in stretch4_urdf.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

BASE_JOINTS = {"x": "base_x_joint", "y": "base_y_joint", "theta": "base_theta_joint"}
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

# Large but finite so the QP limits stay well defined.
_BASE_LIMIT = {"x": 5.0, "y": 5.0, "theta": np.pi}


def _add_virtual_base(urdf_xml: str) -> str:
    """Prepend odom -> base_x -> base_y -> base_theta -> <URDF root>."""
    robot = ET.fromstring(urdf_xml)
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    roots = [
        l.get("name") for l in robot.findall("link") if l.get("name") not in children
    ]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {roots}")

    chain = [
        (
            "base_x_joint",
            "prismatic",
            "1 0 0",
            "mm_ik_odom",
            "mm_ik_base_x",
            _BASE_LIMIT["x"],
        ),
        (
            "base_y_joint",
            "prismatic",
            "0 1 0",
            "mm_ik_base_x",
            "mm_ik_base_y",
            _BASE_LIMIT["y"],
        ),
        (
            "base_theta_joint",
            "revolute",
            "0 0 1",
            "mm_ik_base_y",
            roots[0],
            _BASE_LIMIT["theta"],
        ),
    ]
    for link in ("mm_ik_odom", "mm_ik_base_x", "mm_ik_base_y"):
        ET.SubElement(robot, "link", name=link)
    for name, jtype, axis, parent, child, lim in chain:
        joint = ET.SubElement(robot, "joint", name=name, type=jtype)
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "axis", xyz=axis)
        ET.SubElement(
            joint, "limit", lower=str(-lim), upper=str(lim), effort="100", velocity="1"
        )
    return ET.tostring(robot, encoding="unicode")


def build_model(urdf_xml: str, base_dofs=(), joint_limits=None) -> pin.Model:
    """Return the reduced model with only lift, merged arm, wrist and `base_dofs` free.

    Args:
        urdf_xml: Robot URDF, e.g. the `/robot_description` string.
        base_dofs: Subset of ('x', 'y', 'theta') to leave free.
        joint_limits: Optional {joint_name: (lower, upper)} to tighten limits
            after the arm merge (use MERGED_ARM for the arm). Joints that are not
            free in this model (e.g. base_y_joint with base_dofs=('x',)) are ignored.
    """
    unknown = set(base_dofs) - set(BASE_JOINTS)
    if unknown:
        raise ValueError(
            f"unknown base_dofs {sorted(unknown)}, expected a subset of {list(BASE_JOINTS)}"
        )

    full = pin.buildModelFromXML(_add_virtual_base(urdf_xml))
    missing = [j for j in ARM_SEGMENTS + ARM_JOINTS if not full.existJointName(j)]
    if missing:
        raise ValueError(f"URDF is missing joints {missing}")

    arm_upper = sum(
        full.upperPositionLimit[full.idx_qs[full.getJointId(j)]] for j in ARM_SEGMENTS
    )

    keep = set(ARM_JOINTS) | {BASE_JOINTS[d] for d in base_dofs}
    lock_ids = [full.getJointId(n) for n in full.names[1:] if n not in keep]
    model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))

    model.upperPositionLimit[model.idx_qs[model.getJointId(MERGED_ARM)]] = arm_upper
    for name, (lower, upper) in (joint_limits or {}).items():
        if not model.existJointName(name):
            continue
        idx = model.idx_qs[model.getJointId(name)]
        model.lowerPositionLimit[idx] = max(model.lowerPositionLimit[idx], lower)
        model.upperPositionLimit[idx] = min(model.upperPositionLimit[idx], upper)
    return model
