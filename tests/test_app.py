import os
import pytest
import numpy as np
from plane_stitcher import stitch3D, intersection_over_union
from plane_stitcher.app import intersection_over_union_dense

@pytest.mark.parametrize('filename', [
    ('read_demo_data')
])
def test_stitch3D(filename, request):
    """
    The masks array below has been acwuired as follows:
    masks = io.imread('BZ004_s22_exvivo_0g_cp_masks.tif')
    masks = fastremap.mask_except(masks,[4115, 4356, 4547, 4548, 4756, 4758, 5134, 4949, 5137])
    masks = masks[35:41, 1100:1400, 1400:1700]
    """
    read_demo_data = request.getfixturevalue(filename)
    masks = read_demo_data[0]
    expected = read_demo_data[1]

    out = stitch3D(masks, 0.2)
    assert np.all(out == expected)


@pytest.mark.parametrize('filename', [
    ('read_demo_data')
])
def test_intersection_over_union(filename, request):
    read_demo_data = request.getfixturevalue(filename)
    masks = read_demo_data[0]
    cur = np.vstack([masks[2], masks[1]])
    pred = np.vstack([masks[0], masks[0]])
    cell_area = np.bincount(masks[0].flatten(), minlength=masks[0].shape[1])
    iou = intersection_over_union(cur, pred, cell_area)

    iou_dense_p1 = intersection_over_union_dense(masks[1], masks[0])
    iou_dense_p2 = intersection_over_union_dense(masks[2], masks[0])

    np.allclose(iou.tocsr()[4547, 4356], iou_dense_p1[4547, 4356])
    np.allclose(iou.tocsr()[4548, 4356], iou_dense_p1[4548, 4356])

    np.allclose(iou.tocsr()[4756, 4356], iou_dense_p2[4756, 4356])
    np.allclose(iou.tocsr()[4758, 4356], iou_dense_p2[4758, 4356])

    np.allclose(iou.tocsr()[4547, 4356], 0.4297951582867784)
    np.allclose(iou.tocsr()[4547, 4356], 0.43735035913806863)

    np.allclose(iou.tocsr()[4756, 4356], 0.39081325301204817)
    np.allclose(iou.tocsr()[4758, 4356], 0.45595854922279794)


    assert 1==1

