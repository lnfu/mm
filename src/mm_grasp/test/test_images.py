from types import SimpleNamespace as NS

import numpy as np
import pytest

from mm_grasp.perception.images import (
    decode_mask_png,
    depth_to_array,
    image_to_array,
)


def depth_msg(arr, encoding, pad=0, bigendian=False):
    dtype = arr.dtype.newbyteorder(">" if bigendian else "<")
    rows = arr.astype(dtype).view(np.uint8).reshape(arr.shape[0], -1)
    rows = np.hstack([rows, np.zeros((arr.shape[0], pad), np.uint8)])
    return NS(
        encoding=encoding,
        height=arr.shape[0],
        width=arr.shape[1],
        step=rows.shape[1],
        is_bigendian=bigendian,
        data=rows.tobytes(),
    )


@pytest.mark.parametrize(
    "encoding,dtype", [("16UC1", np.uint16), ("32FC1", np.float32)]
)
@pytest.mark.parametrize("pad", [0, 6])
def test_image_to_array(encoding, dtype, pad):
    arr = (np.arange(12).reshape(3, 4) * 100).astype(dtype)
    assert np.array_equal(image_to_array(depth_msg(arr, encoding, pad)), arr)


def test_image_to_array_bigendian():
    arr = np.array([[1, 513]], np.uint16)
    assert np.array_equal(image_to_array(depth_msg(arr, "16UC1", bigendian=True)), arr)


def test_image_to_array_rejects_color():
    with pytest.raises(ValueError):
        image_to_array(NS(encoding="rgb8"))


def test_decode_mask_png():
    import cv2

    mask = np.zeros((10, 10), np.uint8)
    mask[2:5, 3:8] = 255
    assert decode_mask_png(cv2.imencode(".png", mask)[1].tobytes()).sum() == 15


@pytest.mark.parametrize(
    "fmt", ["16UC1; compressedDepth png", "16UC1; compressedDepth png; seq=3989"]
)
def test_compressed_depth(fmt):
    import cv2

    depth = (np.arange(48).reshape(6, 8) * 37).astype(np.uint16)
    msg = NS(format=fmt, data=bytes(12) + cv2.imencode(".png", depth)[1].tobytes())
    assert np.array_equal(depth_to_array(msg), depth)


def test_compressed_depth_rejects_other_formats():
    with pytest.raises(ValueError):
        depth_to_array(NS(format="16UC1; compressedDepth rvl", data=b""))


def test_depth_to_array_raw():
    arr = np.array([[1, 2]], np.uint16)
    assert np.array_equal(depth_to_array(depth_msg(arr, "16UC1")), arr)
