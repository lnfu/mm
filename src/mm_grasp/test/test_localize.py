import numpy as np
import pytest
from mm_grasp.localize import (
    center_and_width,
    mask_points,
    resize_mask,
    transform_points,
)

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def box_scene(z_obj=0.5, z_bg=1.2, u=(300, 340), v=(220, 260)):
    """Flat object at z_obj in front of a wall at z_bg; returns (depth_mm, mask)."""
    depth = np.full((480, 640), int(z_bg * 1000), np.uint16)
    depth[v[0] : v[1], u[0] : u[1]] = int(z_obj * 1000)
    mask = np.zeros((480, 640), bool)
    mask[v[0] : v[1], u[0] : u[1]] = True
    return depth, mask


def test_center_of_flat_box():
    depth, mask = box_scene()
    pts = mask_points(mask, depth, K)
    center, width = center_and_width(pts, width_axis=0)
    # u in [300, 340) around cx=320 -> x in [-0.02, 0.02) at z=0.5
    assert center == pytest.approx([0.0, 0.0, 0.5], abs=2e-3)
    assert width == pytest.approx(0.04, abs=3e-3)


def test_mask_edge_leaking_background_is_dropped():
    depth, mask = box_scene()
    mask[215:265, 295:345] = True  # mask 5 px larger than the object, sees the wall
    center, _ = center_and_width(mask_points(mask, depth, K))
    assert center[2] == pytest.approx(0.5, abs=1e-3)


def test_float_depth_and_holes():
    depth, mask = box_scene()
    depth = depth.astype(np.float32) / 1000
    depth[230:235, :] = np.nan
    depth[240:245, :] = 0
    center, _ = center_and_width(mask_points(mask, depth, K))
    assert center[2] == pytest.approx(0.5, abs=1e-3)


def test_too_few_points():
    depth, mask = box_scene(u=(300, 303), v=(220, 223))
    with pytest.raises(ValueError):
        mask_points(mask, depth, K)


def test_mask_at_double_resolution():
    depth, mask = box_scene()
    big = np.kron(mask, np.ones((2, 2), bool))
    assert np.array_equal(resize_mask(big, depth.shape), mask)
    center, _ = center_and_width(mask_points(big, depth, K))
    assert center == pytest.approx([0.0, 0.0, 0.5], abs=2e-3)


def test_transform_points():
    # optical frame (z forward, x right, y down) -> a body frame (x forward, y left, z up)
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    pts = transform_points(np.array([[0.1, 0.2, 0.5]]), R, [1.0, 0.0, 0.0])
    assert pts[0] == pytest.approx([1.5, -0.1, -0.2])
