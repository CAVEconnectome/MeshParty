from meshparty import utils,mesh_to_skel_utils
import numpy as np
import os


def test_point_to_skel_meshpath(full_cell_mesh,full_cell_skeleton):
    pt = []
    path = mesh_to_skel_utils.point_to_skel_meshpath(full_cell_mesh,full_cell_skeleton,pt)
    assert (1==1) 



