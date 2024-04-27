import numpy as np
from scipy.sparse import coo_matrix


def split_iou(iou, coo_p2):
    m, n = coo_p2.shape
    p2_labels = np.unique(coo_p2.data)
    idx = np.isin(iou.row, p2_labels)
    upper = coo_matrix((iou.data[idx], (iou.row[idx], iou.col[idx])), shape=iou.shape)
    lower = coo_matrix((iou.data[~idx], (iou.row[~idx], iou.col[~idx])), shape=iou.shape)
    return upper, lower


def shift_labels(mask_1, mask_2):
    coo_1 = coo_matrix(mask_1)
    coo_2 = coo_matrix(mask_2)
    idx = np.in1d(coo_2.data, np.unique(coo_1.data))
    if np.any(idx):
        print('shifted')
        coo_2.data = coo_2.data + coo_1.data.max()
    return coo_1.toarray(), coo_2.toarray()


