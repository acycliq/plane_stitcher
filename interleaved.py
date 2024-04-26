import numpy as np
import fastremap
from plane_stitcher.app import stitch3D

def get_data():
    masks = np.load('/home/dimitris/data/Max/case_interleaved/out_dapi_clean.npz')['arr_0']
    dapi = np.load('/home/dimitris/data/Max/case_interleaved/all_masks_dapi.npz')['im']

    np.save('/home/dimitris/data/Max/case_interleaved/out_dapi_clean.npy', masks)
    np.save('/home/dimitris/data/Max/case_interleaved/dapi.npy', dapi)

    masks_1862_1375 = fastremap.mask_except(masks, [1862, 1375, 523, 587, 589, 590])
    np.save('/home/dimitris/data/Max/case_interleaved/masks_1862_1375.npy', masks_1862_1375[11:, 1469:1949, 1330:1939])

    t = masks_1862_1375 * dapi
    t = t > 0

    out = dapi * t

    np.save('/home/dimitris/data/Max/case_interleaved/dapi_only_1862_1375.npy', out[:, 1469:1949, 1330:1939])

    return out


if __name__ == "__main__":
    # dapi = get_data()
    dapi = np.load('/home/dimitris/data/Max/case_interleaved/dapi_only_1862_1375.npy')
    out = stitch3D(dapi[:])

    np.save('/home/dimitris/data/Max/case_interleaved/out_fixed.npy', out)

    print('Done')

