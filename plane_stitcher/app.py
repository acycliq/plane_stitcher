from numba import jit
import numpy as np
from skimage import io
import fastremap
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix


def intersection_over_union_dense(masks_true, masks_pred):
    """ THE ORIGINAL FUNCTION, TAKEN FROM CELLPOSE, WORKS WITH DENSE ARRAYS"""
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
    """ intersection over union of all mask pairs.
    Taken for cellpose and edited to work with sparse arrays (coo)

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
        # Needs further checking
        mask = overlap.row == 0
        overlap.data[mask] = overlap.data[mask] - adj[np.nonzero(mask)]

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
    """
    You have 3 planes (0,1,2) and you want to get the iou between planes
    (0 and 2) and (0 and 1)
    Stack plane 0 to itself
    Stack plane 2 ontop plane 1
    Do the typical IoU for the two stacked planes
    """

    # 1: stack
    cur = np.vstack([lst[0], lst[1]])
    pred = np.vstack([lst[2], lst[2]])
    # adj = fastremap.unique(lst[2].ravel(), return_counts=True)[1]
    adj = np.bincount(lst[2].flatten(), minlength=lst[2].shape[1])
    iou = intersection_over_union(cur, pred, adj)

    # split the stacked iou to two separates ones.
    # one for each plane
    upper, lower = split_iou(iou, coo_matrix(lst[0]))

    # remove the iou entries for label=0
    upper, lower = remove_label_zero(upper, lower)

    # overwrite the entries in upper (plane 1) with the
    # column-size (nonzero) minimum from lower (plane 0)
    # This is done to make sure that a cell that appears in
    # plane 0 and there is a cell that overlaps with it on plane 1
    # but fails the threshold criterion (hence will not merge) then
    # if on plane 2 there is also another overlapping cell then that
    # cell will not merge (no matter what the overlap is)
    upper = map_min(upper, lower)

    # apply now the threshold
    upper, lower = apply_threshold(upper, lower, stitch_threshold)

    # For each plane keep only the max
    upper = _keep_max(upper)
    lower = _keep_max(lower)

    out = upper.maximum(lower).tocoo()
    return out, cur


def min_col(b):
    """
    reads sparse array b and returns the column ids and the corresponding min
    Example:
        b = coo_matrix([
            [2, 0, 2],
            [0, 0, 1],
            [5, 0, 0],
            [0, 0, 0],
        ])

        col_id, col_min = min_col(b)
        col_id = [0, 2]
        col_min = [2, 1]
    """
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
    """
    replaces the entries in a with the column-wise (non-zero) min of a and b
    Example:
        a = coo_matrix([
            [0, 12, 3],
            [3, 0, 0],
            [0, 0, 1],
            [4, 5, 6],
        ])

        b = coo_matrix([
            [2, 0, 2],
            [0, 0, 1],
            [5, 0, 0],
            [0, 0, 0],
        ])

        c = map_min(a, b)
        c = [
            [0, 12, 1],
            [2, 0, 0],
            [0, 0, 1],
            [2, 5, 1],
]
    """
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


def stitch3D(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    reserved = None

    # add a dummy zero-plane at the very end
    dummy = 0 * masks[0]
    masks = np.concatenate((masks, dummy[None, :, :]))

    for i in range(len(masks) - 2):
        print('stitching plane %d to %d and %d ' % (i, i+1, i+2))
        iou, planes_concat = intersection_over_union_wrapper([masks[i + 2], masks[i + 1], masks[i]], stitch_threshold)
        masks[i+2], masks[i+1], reserved = _stitch_coo(iou, planes_concat, stitch_threshold, mmax, reserved)


    # drop the last plane when you return, it is the dummy plane added
    # at the very beginning
    return masks[:-1].astype(np.uint32)


def remove_label_zero(upper, lower):
    upper = _remove_label_zero(upper)
    lower = _remove_label_zero(lower)
    return upper, lower


def _remove_label_zero(iou_coo):
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
        istitch = iou_coo.argmax(axis=1) + 1
        istitch = np.asarray(istitch).flatten()
        max_axis1 = iou_coo.max(axis=1)
        max_axis1_arr = max_axis1.toarray()
        ino = np.nonzero(max_axis1_arr == 0.0)[0]

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