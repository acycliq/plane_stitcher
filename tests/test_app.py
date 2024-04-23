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


test_list_1 = ([
    [
         1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
             ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0, 0, 0],
        [0, 2, 2, 2, 2, 0, 0],
        [0, 0, 2, 2, 2, 0, 0],
        [0, 0, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
             [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 3, 3, 3]
             ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
)


test_list_2 = ([
    [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
    ],

            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0]
             ],
    [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
)

"""
some basic toy cases:
"""
@pytest.mark.parametrize('test_data', [
    (test_list_1)
])
def test_app(test_data):
    ones = test_list_1[0]
    twos = test_list_1[1]
    threes = test_list_1[2]
    blank = test_list_1[3]
    masks = np.stack([ones, blank, twos, threes])

    # when threshold is 0.19, it should stitch. All cells should get the label=1
    out = stitch3D(masks, stitch_threshold=0.19)

    expected = out > 0
    expected = expected.astype(np.int32)
    assert np.allclose(out, expected)


@pytest.mark.parametrize('test_data', [
    (test_list_1)
])
def test_app(test_data):
    ones = test_list_1[0]
    twos = test_list_1[1]
    threes = test_list_1[2]
    blank = test_list_1[3]
    masks = np.stack([ones, blank, twos, threes])

    # when threshold is 0.19, it should not stitch
    out = stitch3D(masks, stitch_threshold=0.20)

    assert np.allclose(out, masks)


@pytest.mark.parametrize('test_data', [
    (test_list_2)
])
def test_app(test_data):
    big_1 = test_list_1[0]
    small = test_list_1[1]
    big_3 = test_list_1[2]
    masks = np.stack([big_1, small, big_3])

    out = stitch3D(masks)

    assert np.allclose(out, masks)
