import pytest
import numpy as np
import pandas as pd
from .basic_test import full_cell_mesh, full_cell_synapses, full_cell_soma_pt
from meshparty import meshwork, mesh_filters


# Test meshwork creation
@pytest.fixture(scope='session')
def basic_meshwork(full_cell_mesh):
    yield meshwork.load_meshwork('test/test_files/648518346349539862_meshwork.h5')


def test_meshwork_creation(basic_meshwork):
    assert basic_meshwork.seg_id == 648518346349539862
    nrn = meshwork.Meshwork(basic_meshwork.mesh, seg_id=basic_meshwork.seg_id)
    syn_in_df = basic_meshwork.anno.syn_in.data_original

    nrn.add_annotations('syn_in', syn_in_df, point_column='ctr_pt_position')
    assert len(nrn.anno.syn_in.df == nrn.anno['syn_in'].df)
    nrn.add_annotations('syn_in_100', syn_in_df,
                        point_column='ctr_pt_position', max_distance=50)
    assert len(nrn.anno.syn_in_100.df) == 733


def test_meshwork_masking(basic_meshwork):
    nrn = meshwork.Meshwork(basic_meshwork.mesh, seg_id=basic_meshwork.seg_id)
    syn_in_df = basic_meshwork.anno.syn_in.data_original
    nrn.add_annotations('syn_in', syn_in_df, point_column='ctr_pt_position')
    orig_len = len(nrn.anno.syn_in)

    mask = mesh_filters.filter_geodesic_distance(
        nrn.mesh, [287636, 203013], 5000)
    nrn.apply_mask(mask)
    assert len(nrn.anno.syn_in) == 9
    nrn.reset_mask()
    assert len(nrn.anno.syn_in) == orig_len


def test_meshwork_skeleton(basic_meshwork):
    nrn = meshwork.Meshwork(basic_meshwork.mesh, seg_id=basic_meshwork.seg_id,
                            skeleton=basic_meshwork.skeleton)
    assert nrn.skeleton.n_vertices == 11451
    assert nrn.root == 332971
    assert len(nrn.root_region) == 25701

    ds_pts = nrn.downstream_of(287636)
    assert len(ds_pts) == 43411
    assert len(ds_pts.to_skel_index) == 752

    nrn.apply_mask(ds_pts.to_mesh_mask)
    new_mesh = meshwork.Meshwork(nrn.mesh)
    new_mesh.skeletonize_mesh(compute_radius=False)
    assert np.isclose(new_mesh.path_length(), 106050.47)
