from meshparty import utils, mesh_skel_utils, mesh_filters
import numpy as np
import os


def test_point_to_skel_meshpath(full_cell_mesh, full_cell_skeleton):

    pt = [306992, 214288, 11560]
    filterpts = mesh_filters.filter_spatial_distance_from_points(full_cell_mesh, [
                                                                 pt], 2000)
    loc_mesh = full_cell_mesh.apply_mask(filterpts)

    path = mesh_skel_utils.point_to_skel_meshpath(
        loc_mesh, full_cell_skeleton, pt, filterpts)

    assert(len(path) == 16)
    assert(path[0] == 970)
    assert(path[6] == 960)
