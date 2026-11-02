"""The pure modules must not pull in ROS, so the grasp logic stays testable anywhere.

ROS lives in grasp_server, scene, motion/robot and perception/{locator,snapshot}.
A new module goes in PURE unless it talks to ROS.
"""

import subprocess
import sys

PURE = [
    "mm_grasp.config",
    "mm_grasp.errors",
    "mm_grasp.transforms",
    "mm_grasp.skills.grasp",
    "mm_grasp.skills.primitives",
    "mm_grasp.motion.base_control",
    "mm_grasp.motion.planning",
    "mm_grasp.motion.trajectory",
    "mm_grasp.perception.images",
    "mm_grasp.perception.localize",
]

ROS_PREFIXES = ("rclpy", "message_filters", "tf2_ros", "mm_interfaces")


def test_pure_modules_do_not_import_ros():
    # A fresh interpreter: other tests in this run may have imported ROS already.
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in PURE)
        + f"print('\\n'.join(sorted(m for m in sys.modules if m.startswith({ROS_PREFIXES!r})"
        " or m.endswith('_msgs'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == []
