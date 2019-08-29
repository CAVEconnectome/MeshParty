from meshparty import utils,mesh_to_skel_utils
import numpy as np
import os


def test_point_to_skel_meshpath(full_cell_mesh,full_cell_skeleton):
    pt = []
    path = mesh_to_skel_utils.point_to_skel_meshpath(full_cell_mesh,full_cell_skeleton,pt)
    assert (1==1) 

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



