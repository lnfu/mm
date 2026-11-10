from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_manip.config import GoalConfig, StartupConfig
from mm_manip.skills.articulate.config import ArticulateConfig
from mm_manip.skills.grasp.config import GraspConfig

ALL = [StartupConfig, GoalConfig, GraspConfig, ArticulateConfig]


class FakeNode:
    """The two rclpy.Node methods the configs use, backed by a dict."""

    def __init__(self):
        self.params = {}

    def declare_parameter(self, name, value):
        if name in self.params:
            raise ValueError(f"{name} declared twice")
        self.params[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])


@pytest.mark.parametrize("config", ALL)
def test_round_trip_gives_the_defaults(config):
    node = FakeNode()
    config.declare(node)
    assert config.from_node(node) == config()


def test_configs_share_one_node_without_clashes():
    node = FakeNode()
    for config in ALL:
        config.declare(node)
    assert node.params["dry_run"] is True
    assert node.params["grasp.lift_height"] == GraspConfig().lift_height
    assert "lift_height" not in node.params


def test_groups_are_dotted_names():
    node = FakeNode()
    StartupConfig.declare(node)
    GoalConfig.declare(node)
    ArticulateConfig.declare(node)
    assert node.params["frames.ee"] == "grasp_center_link"
    assert node.params["camera.timeout"] == 2.0
    assert node.params["gripper.open"] == 80.0
    assert node.params["articulate.gemini.thinking_level"] == "low"
    assert node.params["articulate.ik.position_error"] == 0.01
    assert node.params["articulate.path.max_velocity.base_x"] == 0.05
    assert "articulate.path.max_velocity" not in node.params


def test_dict_fields_are_one_parameter_per_joint():
    node = FakeNode()
    GraspConfig.declare(node)
    assert node.params["grasp.start_pose.lift_joint"] == 0.0
    assert node.params["grasp.grasp_wrist.wrist_yaw_joint"] == 0.0
    assert "grasp.start_pose" not in node.params


def test_changed_parameters_show_up_in_the_next_read():
    node = FakeNode()
    GraspConfig.declare(node)
    node.params["grasp.start_pose.lift_joint"] = 0.8
    node.params["grasp.base_control.max_linear"] = 0.05
    cfg = GraspConfig.from_node(node)
    assert cfg.start_pose["lift_joint"] == 0.8
    assert cfg.base_control.max_linear == 0.05


def test_config_is_read_only():
    with pytest.raises(AttributeError):
        GraspConfig().lift_height = 0.0


@pytest.mark.parametrize(
    "text, dofs", [("", []), ("x", ["x"]), (" x, y ,theta", ["x", "y", "theta"])]
)
def test_base_dof_list(text, dofs):
    assert GraspConfig(base_dofs=text).base_dof_list == dofs


def test_the_yaml_sets_only_declared_parameters_of_the_right_type():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).parents[1] / "config" / "manipulation.yaml"
    params = yaml.safe_load(path.read_text())["mm_manip"]["ros__parameters"]

    def flatten(tree, prefix=""):
        for key, value in tree.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                yield from flatten(value, name)
            else:
                yield name, value

    node = FakeNode()
    for config in ALL:
        config.declare(node)
    for name, value in flatten(params):
        assert name in node.params, f"{name} is not a parameter"
        default = node.params[name]
        assert isinstance(value, type(default)) or (
            isinstance(default, float) and isinstance(value, int)
        ), f"{name}: {value!r} is not a {type(default).__name__}"
