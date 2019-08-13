import numpy as np
from meshparty import skeleton_quality as sq
import pytest
from skeleton_io_test import full_cell_skeleton
from basic_test import full_cell_mesh, mesh_link_edges



def test_skeleton_quality(full_cell_skeleton, full_cell_mesh, mesh_link_edges):
    sk = full_cell_skeleton
    mesh = full_cell_mesh
    # mesh.link_edges = mesh_link_edges
    pscore, sk_paths, ms_paths, sk_inds_list, mesh_inds_list, path_distances = \
                                        sq.skeleton_path_quality(sk, mesh, return_path_info=True)
    assert len(pscore) == len(sk.cover_paths)
    assert np.isclose(pscore.sum(), 3.9077, 0.001)


