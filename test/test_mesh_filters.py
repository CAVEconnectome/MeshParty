from meshparty import trimesh_io, mesh_filters
import numpy as np
import os
import imageio.v2 as imageio
import pytest


def compare_img_to_test_file(fname, back_val=255, close=15):
    img_test = imageio.imread(fname).astype(np.int16)
    tmpl_path = os.path.join("test/test_files/", os.path.split(fname)[1])
    img_tmpl = imageio.imread(tmpl_path).astype(np.int16)

    non_background = np.any((img_test != back_val) | (img_tmpl != back_val), axis=2)
    diff = np.linalg.norm(img_test - img_tmpl, axis=2)
    perc_close = np.sum((diff < close) & (non_background)) / np.sum(non_background)
    assert perc_close > 0.9


def test_filter_components_by_size(full_cell_mesh):
    only_big = mesh_filters.filter_components_by_size(full_cell_mesh, min_size=5000)
    only_small = mesh_filters.filter_components_by_size(full_cell_mesh, max_size=5000)

    assert np.sum(only_big) == 2260414
    assert np.sum(only_small) == 67154


def test_filter_components_by_dist_From_pts(full_cell_mesh, full_cell_soma_pt):
    pt_down = full_cell_soma_pt + np.array([0, 50000, 0])
    pts = np.vstack([full_cell_soma_pt, pt_down])

    two_pt_mask = mesh_filters.filter_spatial_distance_from_points(
        full_cell_mesh, pts, 10000
    )

    one_pt_mask = mesh_filters.filter_spatial_distance_from_points(
        full_cell_mesh, full_cell_soma_pt, 10000
    )

    far_point = np.array([-100000, -1000000, -100000])

    far_mask = mesh_filters.filter_spatial_distance_from_points(
        full_cell_mesh, far_point, 10000
    )
    assert np.sum(far_mask) == 0
    with pytest.raises(trimesh_io.EmptyMaskException):
        far_mesh = full_cell_mesh.apply_mask(far_mask)


def test_filter_two_points(full_cell_mesh, full_cell_soma_pt, tmp_path):
    pt_down = full_cell_soma_pt + np.array([0, 50000, 0])
    is_large = mesh_filters.filter_largest_component(full_cell_mesh)
    full_cell_mesh = full_cell_mesh.apply_mask(is_large)
    on_ais = mesh_filters.filter_two_point_distance(
        full_cell_mesh, [full_cell_soma_pt, pt_down], 2000
    )

    ais_mesh = full_cell_mesh.apply_mask(on_ais)

    pts_end = np.array([full_cell_soma_pt, pt_down])
    ais_sloppy = mesh_filters.filter_close_to_line(full_cell_mesh, pts_end, 4000)
    assert np.sum(ais_sloppy) == 12961
