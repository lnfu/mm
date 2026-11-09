import subprocess
import sys

# Add each new module that does not use ROS.
PURE = [
    "mm_manip.camera.frame",
    "mm_manip.camera.images",
    "mm_manip.cancel",
    "mm_manip.config",
    "mm_manip.errors",
    "mm_manip.geometry.plane",
    "mm_manip.geometry.points",
    "mm_manip.geometry.transforms",
    "mm_manip.motion.base_control",
    "mm_manip.motion.planning",
    "mm_manip.motion.trajectory",
    "mm_manip.motion.velocity",
    "mm_manip.skills.primitives",
    "mm_manip.skills.articulate.debug",
    "mm_manip.skills.articulate.door",
    "mm_manip.skills.articulate.gemini",
    "mm_manip.skills.articulate.keypoints",
    "mm_manip.skills.articulate.locate",
    "mm_manip.skills.articulate.plan",
    "mm_manip.skills.articulate.skill",
    "mm_manip.skills.grasp.plan",
    "mm_manip.skills.grasp.skill",
]

ROS_PREFIXES = ("rclpy", "message_filters", "tf2_ros", "mm_interfaces")


def test_pure_modules_do_not_import_ros():
    # A new interpreter: other tests can import ROS first.
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
