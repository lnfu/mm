from types import SimpleNamespace

import pytest

from mm_grasp.config import GraspConfig


class FakeNode:
    """The two rclpy.Node methods GraspConfig uses, backed by a dict."""

    def __init__(self):
        self.params = {}

    def declare_parameter(self, name, value):
        if name in self.params:
            raise ValueError(f"{name} declared twice")
        self.params[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])


def test_round_trip_gives_the_defaults():
    node = FakeNode()
    GraspConfig.declare(node)
    assert GraspConfig.from_node(node) == GraspConfig()


def test_dict_fields_are_one_parameter_per_joint():
    node = FakeNode()
    GraspConfig.declare(node)
    assert node.params["start_pose.lift_joint"] == 0.0
    assert node.params["grasp_wrist.wrist_yaw_joint"] == 0.0
    assert "start_pose" not in node.params


def test_changed_parameters_show_up_in_the_next_read():
    node = FakeNode()
    GraspConfig.declare(node)
    node.params["start_pose.lift_joint"] = 0.8
    node.params["lift_height"] = 0.0
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
