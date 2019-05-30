from meshparty import trimesh_io, mesh_filters, trimesh_vtk
import numpy as np
import os 
import imageio

def compare_img_to_test_file(fname, back_val = 255, close=15):
    img_test = imageio.imread(fname).astype(np.int16)
    tmpl_path = os.path.join('test/test_files/', os.path.split(fname)[1])
    img_tmpl = imageio.imread(tmpl_path).astype(np.int16)

    non_background = np.any((img_test != back_val) | (img_tmpl != back_val), axis=2)
    diff = np.linalg.norm(img_test-img_tmpl, axis=2)
    perc_close = np.sum((diff < close) & (non_background))/np.sum(non_background)
    assert(perc_close>.9)

def test_filter_components_by_size(full_cell_mesh):

    only_big = mesh_filters.filter_components_by_size(full_cell_mesh, min_size=5000)
    only_small = mesh_filters.filter_components_by_size(full_cell_mesh, max_size=5000)

    assert(np.sum(only_big)==2260414)
    assert(np.sum(only_small)==67154)


def test_filter_two_points(full_cell_mesh, full_cell_soma_pt, tmp_path):

    pt_down = full_cell_soma_pt + np.array([0,50000,0])
    is_large = mesh_filters.filter_largest_component(full_cell_mesh)
    full_cell_mesh = full_cell_mesh.apply_mask(is_large)
    on_ais = mesh_filters.filter_two_point_distance(full_cell_mesh,
                                                    [full_cell_soma_pt,
                                                     pt_down],
                                                     2000)
    
    ais_mesh = full_cell_mesh.apply_mask(on_ais)
    ais_actor = trimesh_vtk.mesh_actor(ais_mesh)

    fname = 'full_cell_ais.png'
    filepath = os.path.join(tmp_path, fname)

    trimesh_vtk.render_actors([ais_actor],
                                back_color=(1,1,1),
                                do_save=True,
                                filename=filepath,
                                scale=1)
    compare_img_to_test_file(filepath)

    pts_end = np.array([full_cell_soma_pt,pt_down])
    ais_sloppy = mesh_filters.filter_close_to_line(full_cell_mesh,
                                                   pts_end,
                                                   4000)
    assert(np.sum(ais_sloppy)==12961)
