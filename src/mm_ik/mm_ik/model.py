import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from .joints import ARM_SEGMENTS

BASE_JOINTS = {"x": "base_x_joint", "y": "base_y_joint", "theta": "base_theta_joint"}
MERGED_ARM = ARM_SEGMENTS[-1]  # carries the full arm range in the reduced model
ARM_JOINTS = [
    "lift_joint",
    MERGED_ARM,
    "wrist_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
]

# Large but finite, so the QP limits stay defined.
_BASE_LIMIT = {"x": 5.0, "y": 5.0, "theta": np.pi}


def _add_virtual_base(urdf_xml: str) -> str:
    robot = ET.fromstring(urdf_xml)
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    roots = [
        link.get("name")
        for link in robot.findall("link")
        if link.get("name") not in children
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
