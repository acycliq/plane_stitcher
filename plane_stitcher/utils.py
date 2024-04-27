import numpy as np
from scipy.sparse import coo_matrix
import logging

utils_logger = logging.getLogger(__name__)


def shift_labels(mask_1, mask_2):
    """
    merging mask_0 to mask_1 and mask_2 assumes that the two masks (masks_1 and mask_2)
    do not share labels between then (I think they can share labels with mask_0 though...)
    This function will shift those labels in mask_2 if they also appear on mask_1
    """
    coo_1 = coo_matrix(mask_1)
    coo_2 = coo_matrix(mask_2)
    idx = np.in1d(coo_2.data, np.unique(coo_1.data))
    if np.any(idx):
        shared = coo_2.data[idx]
        new_labels = shared + coo_1.data.max()
        utils_logger.info('Shifted labels %s to %s' % (np.unique(shared), np.unique(new_labels)))
        coo_2.data[idx] = new_labels
    return coo_1.toarray(), coo_2.toarray()


def split_iou(iou, p01):
    """
    splits the stacked overlap to two ious, one for each individual plane.
    If you stitch plane_0 to plane_1 and plane_2, the returned values, lower and upper
    are the corresponding ious
    The arg p01 is needed only to get which cells overlap with which between
    plane_0 and plane_1. It looks a bit stupid because typically p01 is paased in
    as an overlap array (in sparse datatype) hence why bothered with stacking planes ontop
    of each other??
    The stitching logic from cellpose, later on will shift the labels of the non-overlapping
    cells later on and I wanted to make sure that the labels in both plane_1 and plane_2 are
    shifted at the same time and not separately to avoid unforseen errors
    """
    coo = coo_matrix(p01)
    coords = list(zip(coo.row, coo.col))
    has_zero = [~np.all(d) for d in coords]
    coo.data[has_zero] = 0
    coo.eliminate_zeros()

    iou_rc = list(zip(iou.row, iou.col))
    coords = list(zip(coo.row, coo.col))
    idx_1 = np.array([d in set(coords) for d in iou_rc])

    lower = coo_matrix((iou.data[idx_1], (iou.row[idx_1], iou.col[idx_1])), shape=iou.shape)
    upper = coo_matrix((iou.data[~idx_1], (iou.row[~idx_1], iou.col[~idx_1])), shape=iou.shape)

    return upper, lower


def remove_label_zero(upper, lower):
    upper = _remove_label_zero(upper)
    lower = _remove_label_zero(lower)
    return upper, lower


def _remove_label_zero(iou_coo):
    """
    Basically it purges an array by dropping the top row ane and first column. The express
    the iou of those cells that were previously on the background, or the other way round, ie
    cell that have disappeared, hence their footprint now is on the background
    """
    # remove now first column and first row from the coo matrix. These entries correspond to label=0
    is_coord_zero = iou_coo.row * iou_coo.col
    row = iou_coo.row[is_coord_zero != 0] - 1
    col = iou_coo.col[is_coord_zero != 0] - 1
    data = iou_coo.data[is_coord_zero != 0]
    m, n = iou_coo.shape
    return coo_matrix((data, (row, col)), shape=(m - 1, n - 1))


