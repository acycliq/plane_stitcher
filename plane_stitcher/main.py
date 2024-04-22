from scipy.ndimage import generate_binary_structure
from numba import jit
import pandas as pd
import numpy as np
from skimage import morphology
from skimage import io
from skimage.measure import regionprops_table
import fastremap
from scipy import ndimage
import diplib as dip
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix


def intersection_over_union_dense(masks_true, masks_pred):
    """ intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """

    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def intersection_over_union(masks_true, masks_pred, adj=None):
    """ intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap2(masks_true, masks_pred)

    # adjust
    if adj is not None:
        mask = overlap.row == 0
        overlap.data[mask] = overlap.data[mask] - adj


    # print('x.max is %d' % masks_true.max())
    # print('y.max is %d' % masks_pred.max())
    n_pixels_pred = overlap.sum(axis=0)
    n_pixels_true = overlap.sum(axis=1)

    x = np.array((n_pixels_true)).flatten()
    y = np.array((n_pixels_pred)).flatten()

    pred_plus_true = x[overlap.row] + y[overlap.col]
    iou_data = overlap.data /(pred_plus_true - overlap.data)
    iou = coo_matrix((iou_data, (overlap.row, overlap.col)), shape=overlap.shape)

    return iou


@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


# @jit(nopython=True)
def _label_overlap2(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()
    data = np.ones(len(x))
    n_rows = 1 + x.max().astype(np.int32)
    n_cols = 1 + y.max().astype(np.int32)
    overlap = coo_matrix((data, (x, y)), shape = (n_rows, n_cols), dtype=np.int32)
    # overlap = coo_matrix((1 + x.max().astype(np.int32), 1 + y.max().astype(np.int32)), dtype=np.int32 )
    # overlap.data[x, y] += 1
    overlap = overlap.tocsc().tocoo()
    return overlap


def _stitch(iou, mask, stitch_threshold, mmax):
    if iou.size > 0:
        iou[iou < stitch_threshold] = 0.0
        iou[iou < iou.max(axis=0)] = 0.0  # Keeps the max
        istitch = iou.argmax(axis=1) + 1
        ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
        istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, int)
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)
        return istitch[mask]


def intersection_over_union_wrapper(lst, stitch_threshold):
    cur = np.vstack([lst[0], lst[1]])
    pred = np.vstack([lst[2], lst[2]])
    adj = fastremap.unique(lst[2].ravel(), return_counts=True)[1]
    iou = intersection_over_union(cur, pred, adj)

    upper, lower = split_iou(iou, coo_matrix(lst[0]))
    upper, lower = remove_label_zero(upper, lower)

    upper = map_min(upper, lower)
    upper, lower = apply_threshold(upper, lower, stitch_threshold)

    upper = _keep_max(upper)
    lower = _keep_max(lower)


    out = upper.maximum(lower).tocoo()
    return out, cur


def min_col(b):
    b = b.tocsc()
    indptr = b.tocsc().indptr
    indices = b.tocsc().indices
    b_data = b.tocsc().data

    col_id = []
    col_min = []
    for i in range(len(indptr) - 1):
        k = indptr[i]
        l = indptr[i + 1]
        row = b_data[k:l]
        if len(row) > 0:
            col_id.append(i)
            col_min.append(min(row))
    return col_id, col_min


def map_min(a, b):
    col_id, col_min = min_col(b)
    for i, v in enumerate(col_id):
        mask = a.col == v
        a.data[mask] = col_min[i]
    return a


def split_iou(iou, coo_p2):
    m, n = coo_p2.shape
    p2_labels = np.unique(coo_p2.data)
    idx = np.isin(iou.row, p2_labels)
    upper = coo_matrix((iou.data[idx], (iou.row[idx], iou.col[idx])), shape=iou.shape)
    lower = coo_matrix((iou.data[~idx], (iou.row[~idx], iou.col[~idx])), shape=iou.shape)
    return upper, lower


def apply_threshold(upper, lower, stitch_threshold):
    upper = _apply_threshold(upper, stitch_threshold)
    lower = _apply_threshold(lower, stitch_threshold)
    return upper, lower

def _apply_threshold(iou, stitch_threshold):
    idx = iou.data >= stitch_threshold
    iou.data = iou.data[idx]
    iou.col = iou.col[idx]
    iou.row = iou.row[idx]
    return iou


def maximum_iou(lst):
    cur = lst[0].tocsr()
    for i in range(len(lst)-1):
        nxt = lst[i+1].tocsr()
        cur = cur.maximum(nxt)
    return cur.tocoo()


def stitch3D_coo(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    mapped_to_labels = None

    # add a dummy zero-plane at the very end
    dummy = 0 * masks[0]
    masks = np.concatenate((masks, dummy[None, :, :]))

    for i in range(len(masks) - 2):
        print('stitching plane %d to %d and %d ' % (i, i+1, i+2))
        iou, planes_concat = intersection_over_union_wrapper([masks[i + 2], masks[i + 1], masks[i]], stitch_threshold)
        print(iou.toarray())
        masks[i+2], masks[i+1], mapped_to_labels = _stitch_coo(iou, planes_concat, stitch_threshold, mmax, mapped_to_labels)
        print(masks)
        print('====================================')



    # drop the last plane when you return, it is the dummy plane added
    # at the very beginning
    return masks[:-1].astype(np.uint32)


def remove_label_zero(upper, lower):
    upper = _remove_label_zero(upper)
    lower = _remove_label_zero(lower)
    return upper, lower


def _remove_label_zero(iou_coo):
    # remove now first column and first row from the coo matrix. These entries correspond to label=0
    # remove now first column and first row from the coo matrix. These entries correspond to label=0
    is_coord_zero = iou_coo.row * iou_coo.col
    row = iou_coo.row[is_coord_zero != 0] - 1
    col = iou_coo.col[is_coord_zero != 0] - 1
    data = iou_coo.data[is_coord_zero != 0]
    m, n = iou_coo.shape
    return coo_matrix((data, (row, col)), shape=(m - 1, n - 1))


def _stitch_coo(iou_coo, mask, stitch_threshold, mmax, reserved_labels=None):
    out = mask
    m, n = out.shape
    m = int(m / 2)

    if iou_coo.data.size > 0:

        # iou_coo = _keep_max(iou_coo)
        print('Relabelling (old, new)')
        print(*[d for d in zip( iou_coo.row+1, iou_coo.col+1)], sep='\n')
        # iou = iou_coo.toarray()
        # iou[iou < iou.max(axis=0)] = 0.0
        # iou_coo = coo_matrix(iou, shape=iou_coo.shape)

        istitch = iou_coo.argmax(axis=1) + 1
        istitch = np.asarray(istitch).flatten()
        max_axis1 = iou_coo.max(axis=1)
        max_axis1_arr = max_axis1.toarray()
        ino = np.nonzero(max_axis1_arr == 0.0)[0]
        # ino = np.setdiff1d(ino, reserved_labels)

        print('Shifting labels: ')
        print(*ino.tolist(), sep='\t')
        istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)

        if reserved_labels is not None:
            # do not shift those labels
            istitch[reserved_labels] = reserved_labels
        out = istitch[mask]

    return out[:m, :], out[m:, :], iou_coo.col+1


def _keep_max(coo):
    """
    This is the same as:
        iou[iou < iou.max(axis=0)] = 0.0
    where iou is the dense array of coo
    """
    out = coo
    if len(coo.data) > 0:
        csr = coo.tocsr()
        coo_data = coo.data
        column_max = csr.max(axis=0).data
        mask = ~np.in1d(coo.data, column_max)
        coo_data[mask] = 0
        out = coo_matrix((coo_data, (coo.row, coo.col)), shape=coo.shape)
    return out




if __name__ == "__main__":

    ones = [
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]

    twos = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0, 0, 0],
        [0, 2, 2, 2, 2, 0, 0],
        [0, 0, 2, 2, 2, 0, 0],
        [0, 0, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]

    threes = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 3, 3, 3]
    ]

    blank = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]

    sixs_and_threes = [
        [6, 6, 6, 0, 0, 0, 0],
        [6, 6, 6, 0, 0, 0, 0],
        [6, 6, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 3, 3, 3]
    ]
    masks = np.stack([ones, blank, twos, threes])

    dummy = np.zeros(masks[0].shape)
    my_dummy = 0 * masks[0]
    # masks = np.concatenate((masks, np.array(blank)[None,:,:]))
    # masks = np.concatenate((masks, my_dummy[None, :, :]))
    # out = stitch3D_coo(masks, stitch_threshold=0.19)

    zmin = 10
    zmax = 15
    ymin = 30 + 13
    ymax = 70
    xmin = 10 + 10
    xmax = 50

    masks = np.stack([ones, blank, twos, threes], 0.14)
    # masks = np.stack([ones, threes, sixs_and_threes])
    # out = stitch3D_coo(masks, 0.2)

    masks = io.imread('/home/dimitris/data/Max/BZ004_s22_exvivo_0g_cp_masks.tif')
    # # np.save('/home/dimitris/data/Max/masks_roi.npy', masks[37:52, 1100:1400, 1400:1700])
    masks = fastremap.mask_except(masks,[4115, 4356, 4547, 4548, 4756, 4758, 5134, 4949, 5137])  # , 5137, 5301, 5482, 5692])
    #
    # REMOVE 4949 and it stitches
    # KEEP 4949 and it doesnt stitch
    # WHY???
    # masks = masks[37:40, 1100:1400, 1400:1700]
    out = stitch3D_coo(masks[35:41, 1100:1400, 1400:1700], 0.2)
    # # out = skimage.color.label2rgb(out, bg_label=0)
    np.save('/home/dimitris/data/Max/case_1349/out_testing.npy', out)
    print('=========================================')
    print(out)