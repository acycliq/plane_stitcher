import os
import numpy as np
import pytest


@pytest.fixture
def read_demo_data():
    masks_in = np.load('/home/dimitris/dev/python/plane_stitcher/tests/data/masks_in.npy')
    masks_out = np.load('/home/dimitris/dev/python/plane_stitcher/tests/data/masks_out.npy')
    return masks_in, masks_out