from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RawDetection:
    label: str
    score: float
    bbox: tuple  # (x_min, y_min, x_max, y_max) in pixels
    mask: np.ndarray  # bool, same size as the source image


def parse_prompt(prompt: str) -> list:
    """'cup, red bottle' -> ['cup', 'red bottle']."""
    return [p.strip() for p in prompt.split(",") if p.strip()]


def to_detections(boxes_xyxy, scores, class_ids, masks, names, min_score) -> list:
    """Filter by `min_score` and sort by score, highest first.

    Args:
        boxes_xyxy: (N, 4) float array.
        scores: (N,) float array.
        class_ids: (N,) int array, indices into `names`.
        masks: (N, H, W) array at source image size, nonzero = object.
        names: class names, indexed by class id.
    """
    dets = []
    for box, score, cls, mask in zip(boxes_xyxy, scores, class_ids, masks):
        if score < min_score:
            continue
        x0, y0, x1, y1 = (round(v) for v in box)
        dets.append(
            RawDetection(
                names[int(cls)], float(score), (x0, y0, x1, y1), np.asarray(mask) > 0.5
            )
        )
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def encode_mask_png(mask: np.ndarray) -> bytes:
    """bool mask -> mono8 PNG bytes (255 = object)."""
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
