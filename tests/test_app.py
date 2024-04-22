import os
import pytest
import numpy as np
from plane_stitcher import stitch3D

@pytest.mark.parametrize('filename', [
    ('read_demo_data')
])
def test_stitch3D(filename, request):
    read_demo_data = request.getfixturevalue(filename)
    masks = read_demo_data[0]
    expected = read_demo_data[1]

    out = stitch3D(masks, 0.2)
    assert np.all(out == expected)
