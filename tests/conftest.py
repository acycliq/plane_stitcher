import requests
import numpy as np
import pytest


@pytest.fixture
def read_demo_data():
    # masks_in_url = 'https://github.com/acycliq/plane_stitcher/raw/master/tests/data/masks_in.npy'
    # r_in = requests.get(masks_in_url)
    # f = open('masks_in.npy', 'wb')
    # f.write(r_in.content)
    # f.close()

    masks_in = np.load('./data/masks_in.npy')
    masks_out = np.load('./data/masks_out.npy')

    return masks_in, masks_out
