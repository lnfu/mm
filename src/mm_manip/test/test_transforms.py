from types import SimpleNamespace as NS

import numpy as np
import pytest

from mm_manip.transforms import quaternion_to_matrix, transform_to_rt


def test_quaternion_to_matrix():
    s = np.sqrt(0.5)  # 90 deg about z
    R = quaternion_to_matrix(NS(x=0, y=0, z=s, w=s))
    assert R @ [1, 0, 0] == pytest.approx([0, 1, 0])


def test_transform_to_rt():
    s = np.sqrt(0.5)
    R, t = transform_to_rt(
        NS(translation=NS(x=1.0, y=0.0, z=0.5), rotation=NS(x=0, y=0, z=s, w=s))
    )
    assert R @ [0.2, 0, 0] + t == pytest.approx([1.0, 0.2, 0.5])
