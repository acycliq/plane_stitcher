`pip install --force-reinstall  git+https://github.com/acycliq/plane_stitcher.git`

    import plane_stitcher as ps
    import numpy as np

    masks = io.imread('masks.tif')
    masks = ps.stitch3D(masks, 0.2) # default value is 0.25
    masks = ps.fill_holes_and_remove_small_masks(masks, 20) # default value is 15


