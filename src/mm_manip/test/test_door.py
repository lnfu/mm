import json

import cv2
import numpy as np
import pytest

from mm_manip.perception.door import (
    CORNERS,
    Door,
    DoorKeypoints,
    DoorType,
    GeminiDoorKeypoints,
    HandleType,
    build_prompt,
    encode_jpeg,
    fit_plane,
    intersect_plane,
    locate_door,
    parse_response,
    pixel_rays,
    ransac_iterations,
)

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
W, H = 640, 480


def answer(**overrides):
    data = {
        "type": "revolute_left",
        "handle_type": "bar_vertical",
        "handle": [500, 750],
        "top_left": [100, 250],
        "top_right": [100, 750],
        "bottom_left": [900, 250],
        "bottom_right": [900, 750],
    }
    data.update(overrides)
    return json.dumps(data)


def test_prompt_names_the_target():
    assert "the top-left cabinet door" in build_prompt("the top-left cabinet door")


def test_parse_scales_y_x_to_pixels():
    keypoints = parse_response(answer(), W, H)
    assert keypoints.type == DoorType.REVOLUTE_LEFT
    assert keypoints.handle_type == HandleType.BAR_VERTICAL
    assert keypoints.handle == pytest.approx((0.75 * W, 0.5 * H))
    assert keypoints.corners["top_left"] == pytest.approx((0.25 * W, 0.1 * H))
    assert set(keypoints.corners) == set(CORNERS)


def test_parse_accepts_a_fenced_answer():
    keypoints = parse_response(f"Here:\n```json\n{answer()}\n```", W, H)
    assert keypoints.handle == pytest.approx((0.75 * W, 0.5 * H))


@pytest.mark.parametrize(
    "name, door_type",
    [
        ("revolute_right", DoorType.REVOLUTE_RIGHT),
        ("prismatic", DoorType.PRISMATIC),
        ("unknown", DoorType.UNKNOWN),
    ],
)
def test_parse_door_types(name, door_type):
    assert parse_response(answer(type=name), W, H).type == door_type


@pytest.mark.parametrize(
    "text",
    [
        "no idea",
        "[1, 2]",
        answer(type="sliding"),
        answer(handle_type="lever"),
        answer(handle=[1]),
        answer(top_left="here"),
        json.dumps({"type": "prismatic", "handle": [1, 2]}),  # no corners
    ],
)
def test_parse_rejects_other_answers(text):
    with pytest.raises(ValueError):
        parse_response(text, W, H)


def test_door_types_match_the_action():
    # Articulate.action result: TYPE_UNKNOWN=0, REVOLUTE_LEFT=1, REVOLUTE_RIGHT=2, PRISMATIC=3
    assert [int(t) for t in DoorType] == [0, 1, 2, 3]
    # HANDLE_UNKNOWN=0, BAR_VERTICAL=1, BAR_HORIZONTAL=2, KNOB=3, RECESSED=4
    assert [int(h) for h in HandleType] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "name, handle_type",
    [("knob", HandleType.KNOB), ("bar_horizontal", HandleType.BAR_HORIZONTAL)],
)
def test_parse_handle_types(name, handle_type):
    assert parse_response(answer(handle_type=name), W, H).handle_type == handle_type


def test_fit_plane_ignores_a_protruding_handle():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.2, 0.2, size=(400, 2))
    tilt = np.deg2rad(20)
    # Plane through (0, 0, 0.8), tilted about the camera y axis.
    z = 0.8 + xy[:, 0] * np.tan(tilt)
    points = np.column_stack((xy, z))
    points[:40, 2] -= 0.04  # a handle 4 cm in front of the door
    normal, d = fit_plane(points)
    expected = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    assert normal == pytest.approx(expected, abs=1e-3)
    assert d < 0  # faces the camera
    assert normal @ np.array([0, 0, 0.8]) == pytest.approx(d, abs=1e-3)


def test_fit_plane_keeps_the_door_when_40_percent_is_another_panel():
    rng = np.random.default_rng(1)
    door = np.column_stack((rng.uniform(-0.2, 0.2, (600, 2)), np.full(600, 0.8)))
    door[:, 2] += rng.normal(0, 0.003, 600)  # depth noise
    # A side panel, perpendicular to the door, inside a loose quad.
    side = np.column_stack(
        (np.full(400, 0.2), rng.uniform(-0.2, 0.2, 400), rng.uniform(0.6, 0.8, 400))
    )
    normal, d = fit_plane(np.vstack((door, side)))
    assert normal == pytest.approx([0, 0, -1], abs=1e-2)
    assert d == pytest.approx(-0.8, abs=2e-3)


def test_fit_plane_is_repeatable():
    rng = np.random.default_rng(2)
    points = np.column_stack((rng.uniform(-1, 1, (300, 2)), rng.normal(1, 0.05, 300)))
    a, b = fit_plane(points), fit_plane(points)
    assert a[0] == pytest.approx(b[0]) and a[1] == pytest.approx(b[1])


def test_fit_plane_needs_three_points():
    with pytest.raises(ValueError):
        fit_plane(np.zeros((2, 3)))


@pytest.mark.parametrize("ratio, tries", [(1.0, 1), (0.7, 17), (0.4, 105), (0.0, 500)])
def test_ransac_iterations(ratio, tries):
    assert ransac_iterations(ratio, 0.999, 500) == tries


def test_rays_hit_the_plane_where_expected():
    normal, d = np.array([0.0, 0.0, -1.0]), -0.8  # z = 0.8
    hit = intersect_plane(pixel_rays([(420.0, 240.0)], K), normal, d)
    assert hit[0] == pytest.approx([0.16, 0.0, 0.8])


def test_plane_behind_the_camera_raises():
    with pytest.raises(ValueError):
        intersect_plane(pixel_rays([(320.0, 240.0)], K), np.array([0, 0, -1.0]), 0.8)


def wall_scene(z_door=0.8, z_bg=2.0):
    """A door panel at z_door covering u 160..480, v 48..432; background behind."""
    depth = np.full((H, W), int(z_bg * 1000), np.uint16)
    depth[48:432, 160:480] = int(z_door * 1000)
    corners = {
        "top_left": (160.0, 48.0),
        "top_right": (480.0, 48.0),
        "bottom_left": (160.0, 432.0),
        "bottom_right": (480.0, 432.0),
    }
    return depth, DoorKeypoints(
        DoorType.REVOLUTE_LEFT, HandleType.KNOB, (440.0, 240.0), corners
    )


def test_locate_door_puts_corners_on_the_panel_despite_edge_depth():
    depth, keypoints = wall_scene()
    depth[:, 158:163] = 0  # the edges read nothing, as at real depth edges
    door = locate_door(keypoints, depth, K)
    assert door.type == DoorType.REVOLUTE_LEFT
    assert door.handle == pytest.approx([0.192, 0.0, 0.8], abs=2e-3)
    assert door.corners["top_left"] == pytest.approx([-0.256, -0.3072, 0.8], abs=2e-3)
    assert door.corners["bottom_right"] == pytest.approx([0.256, 0.3072, 0.8], abs=2e-3)
    assert door.normal == pytest.approx([0, 0, -1], abs=1e-3)


def test_locate_door_without_depth_raises():
    depth, keypoints = wall_scene()
    depth[:] = 0
    with pytest.raises(ValueError):
        locate_door(keypoints, depth, K)


def test_door_transform():
    door = Door(
        DoorType.REVOLUTE_RIGHT,
        HandleType.KNOB,
        np.array([1.0, 0, 0]),
        {n: np.zeros(3) for n in CORNERS},
        np.array([0, 0, 1.0]),
    )
    rotation = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])  # 90 deg about z
    moved = door.transform(rotation, [0, 0, 1.0])
    assert moved.handle == pytest.approx([0, 1.0, 1.0])
    assert moved.corners["top_left"] == pytest.approx([0, 0, 1.0])
    assert moved.normal == pytest.approx([0, 0, 1.0])  # direction: not translated
    assert moved.type == DoorType.REVOLUTE_RIGHT
    assert moved.handle_type == HandleType.KNOB


def test_encode_jpeg_shrinks_to_max_side():
    image = np.zeros((480, 640, 3), np.uint8)
    decoded = cv2.imdecode(np.frombuffer(encode_jpeg(image, 320), np.uint8), 1)
    assert decoded.shape == (240, 320, 3)
    decoded = cv2.imdecode(np.frombuffer(encode_jpeg(image, 1000), np.uint8), 1)
    assert decoded.shape == (480, 640, 3)  # never enlarged


class FakeGemini:
    def __init__(self, text):
        self.text = text
        self.asked = []

    def ask(self, image, mime_type, prompt, should_cancel):
        self.asked.append((image, mime_type, prompt))
        return self.text


def test_gemini_keypoints_are_in_the_requested_size():
    client = FakeGemini(answer())
    source = GeminiDoorKeypoints(client, max_side=320)
    image = np.zeros((480, 640, 3), np.uint8)
    keypoints = source.keypoints(image, "the left door", (1280, 720), lambda: False)
    assert keypoints.handle == pytest.approx((0.75 * 1280, 0.5 * 720))
    _, mime_type, prompt = client.asked[0]
    assert mime_type == "image/jpeg" and "the left door" in prompt
