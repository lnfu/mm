from types import SimpleNamespace

import pytest

from mm_manip.config import GraspConfig, ServerConfig


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


@pytest.mark.parametrize("config", [ServerConfig, GraspConfig])
def test_round_trip_gives_the_defaults(config):
    node = FakeNode()
    config.declare(node)
    assert config.from_node(node) == config()


def test_configs_share_one_node_without_clashes():
    node = FakeNode()
    ServerConfig.declare(node)
    GraspConfig.declare(node)
    assert node.params["dry_run"] is True
    assert node.params["grasp.lift_height"] == GraspConfig().lift_height
    assert "lift_height" not in node.params


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
    node.params["grasp.lift_height"] = 0.0
    cfg = GraspConfig.from_node(node)
    assert cfg.start_pose["lift_joint"] == 0.8
    assert cfg.lift_height == 0.0


def test_config_is_read_only():
    with pytest.raises(AttributeError):
        GraspConfig().lift_height = 0.0


@pytest.mark.parametrize(
    "text, dofs", [("", []), ("x", ["x"]), (" x, y ,theta", ["x", "y", "theta"])]
)
def test_base_dof_list(text, dofs):
    assert GraspConfig(base_dofs=text).base_dof_list == dofs
