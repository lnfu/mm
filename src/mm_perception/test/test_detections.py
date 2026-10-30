import cv2
import numpy as np
from mm_perception.detections import encode_mask_png, parse_prompt, to_detections


def test_parse_prompt():
    assert parse_prompt("cup") == ["cup"]
    assert parse_prompt(" cup, red bottle ,, ") == ["cup", "red bottle"]
    assert parse_prompt("  ") == []


def test_filter_and_sort():
    masks = np.zeros((3, 4, 5), np.float32)
    masks[1, 1:3, 1:4] = 1.0
    dets = to_detections(
        boxes_xyxy=np.array([[0, 0, 1, 1], [1.4, 1.6, 3.5, 2.9], [0, 0, 2, 2]]),
        scores=np.array([0.2, 0.9, 0.5]),
        class_ids=np.array([0, 1, 0]),
        masks=masks,
        names=["cup", "bottle"],
        min_score=0.3,
    )
    assert [d.label for d in dets] == ["bottle", "cup"]
    assert dets[0].bbox == (1, 2, 4, 3)
    assert dets[0].mask.dtype == bool and dets[0].mask.sum() == 6


def test_mask_png_round_trip():
    mask = np.zeros((48, 64), bool)
    mask[10:20, 5:40] = True
    decoded = cv2.imdecode(
        np.frombuffer(encode_mask_png(mask), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    assert np.array_equal(decoded == 255, mask)
