from types import SimpleNamespace as NS

import numpy as np
import pytest

from mm_manip.geometry.transforms import transform_to_se3


def test_transform_to_se3():
    s = np.sqrt(0.5)  # 90 deg about z
    pose = transform_to_se3(
        NS(translation=NS(x=1.0, y=0.0, z=0.5), rotation=NS(x=0, y=0, z=s, w=s))
    )
    assert pose.rotation @ [1, 0, 0] == pytest.approx([0, 1, 0])
    assert pose.act(np.array([0.2, 0, 0])) == pytest.approx([1.0, 0.2, 0.5])
