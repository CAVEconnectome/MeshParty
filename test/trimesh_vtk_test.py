from meshparty import trimesh_io, trimesh_vtk, skeleton_io
import contextlib
import numpy as np
import pytest
import os
import imageio
import json 

def compare_img_to_test_file(fname, back_val = 255, close=15):
    img_test = imageio.imread(fname)
    tmpl_path = os.path.join('test/test_files/', os.path.split(fname)[1])
    img_tmpl = imageio.imread(tmpl_path)
    assert(img_test.shape == img_tmpl.shape)

    non_background = np.any((img_test != back_val) | (img_tmpl != back_val), axis=2)
    newshape = (img_test.shape[0]*img_test.shape[1], img_test.shape[2])
    img_test_non_back = img_test.reshape(newshape)[non_background.ravel(),:]
    img_tmpl_non_back = img_tmpl.reshape(newshape)[non_background.ravel(),:]

    assert(np.all((np.mean(img_test_non_back, axis=0)- np.mean(img_tmpl_non_back, axis=0)) < close))
    assert(np.all((np.std(img_test_non_back, axis=0)- np.std(img_tmpl_non_back, axis=0)) < close))
    return True

def eval_actor_image(actors, fname, tmp_path, camera=None, scale=2, make_image=False):
    filepath = os.path.join(tmp_path, fname)

    if make_image:
        fpath = fname
    else:
        fpath = filepath
    trimesh_vtk.render_actors(actors, do_save =True,
                              scale=scale,
                              camera=camera,
                              filename=fpath,
                              back_color=(1,1,1))
    if make_image:
        return True
    else:
        return compare_img_to_test_file(filepath)    

@contextlib.contextmanager
def build_basic_mesh():
    verts = np.array([[-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5,  0.5,  0.5]], dtype=np.float32)
    faces = np.array([[ 0,  1,  2],
        [ 3,  2,  1],
        [ 4,  6,  5],
        [ 7,  5,  6],
        [ 8, 10,  9],
        [11,  9, 10],
        [12, 13, 14],
        [15, 14, 13],
        [16, 18, 17],
        [19, 17, 18],
        [20, 21, 22],
        [23, 22, 21]], np.uint32)
   
    yield verts, faces


@contextlib.contextmanager
def build_full_cell_skeleton():
    filepath = 'test/test_files/sk_648518346349499581.h5'
    skel = skeleton_io.read_skeleton_h5(filepath)
    yield skel


@pytest.fixture(scope='session')
def cube_verts_faces():
    with build_basic_mesh() as m:
        yield m

@pytest.fixture(scope='session')
def cell_skel():
    with build_full_cell_skeleton() as sk:
        yield sk


def test_basic_mesh_actor(cube_verts_faces):
    verts, faces = cube_verts_faces
    mesh = trimesh_io.Mesh(verts, faces, process=False)
    mesh_actor = trimesh_vtk.mesh_actor(mesh)
    mesh_actor.GetMapper().Update()
    pd = mesh_actor.GetMapper().GetInput()
    verts_out, faces_out, tmp = trimesh_vtk.poly_to_mesh_components(pd)

    assert(np.all(verts == verts_out))
    assert(np.all(faces_out == faces))
    

def test_skeleton_viz(cell_skel, tmp_path):
    skel_actor = trimesh_vtk.skeleton_actor(cell_skel, vertex_property='rs', line_width=5)
    pd = skel_actor.GetMapper().GetInput()
    verts_out, faces_out, edges_out = trimesh_vtk.poly_to_mesh_components(pd)
    assert(np.all(cell_skel.vertices == verts_out))
    assert(np.all(cell_skel.edges == edges_out))

    eval_actor_image([skel_actor], 'test_sk_render.png', tmp_path, scale=1)


def test_full_cell_camera(full_cell_mesh, full_cell_soma_pt, tmp_path):
    mesh_actor = trimesh_vtk.mesh_actor(full_cell_mesh)
    camera = trimesh_vtk.oriented_camera(full_cell_soma_pt, backoff=100)
    eval_actor_image([mesh_actor], 'full_cell_orient_camera.png', tmp_path, camera=camera, scale=1)

def test_vtk_errors():
    verts = np.random.rand(10,3)
    tris = np.random.randint(0,10,(5,3))
    with pytest.raises(ValueError) as e:
        pd=trimesh_vtk.graph_to_vtk(verts,tris)

    bad_edges = np.arange(0,12).reshape(6,2)
    with pytest.raises(ValueError) as e:
        pd=trimesh_vtk.graph_to_vtk(verts,bad_edges)

    quads = np.random.randint(0,10,(5,4))
    with pytest.raises(ValueError) as e:
        pd = trimesh_vtk.trimesh_to_vtk(verts, quads)

    bad_tris = np.arange(0,12).reshape(4,3)
    with pytest.raises(ValueError) as e:
        pd = trimesh_vtk.trimesh_to_vtk(verts, bad_tris)

def test_full_cell_with_links(full_cell_mesh, full_cell_merge_log, tmp_path, monkeypatch):

    class MyChunkedGraph(object):
        def __init__(a, **kwargs):
            pass

        def get_merge_log(self, atomic_id):    
            return full_cell_merge_log

    monkeypatch.setattr(trimesh_io.trimesh_repair.chunkedgraph,
                        'ChunkedGraphClient',
                        MyChunkedGraph)

    full_cell_mesh.add_link_edges('test', 5)

    mesh_actor = trimesh_vtk.mesh_actor(full_cell_mesh)
    eval_actor_image([mesh_actor], 'full_cell_with_links.png', tmp_path, scale=1)

    mesh_actor = trimesh_vtk.mesh_actor(full_cell_mesh,
                                        opacity=1.0,
                                        show_link_edges = True)

    m1  =np.array(full_cell_merge_log['merge_edge_coords'][0]) 
    ctr = np.mean(m1, axis=0)
    camera = trimesh_vtk.oriented_camera(ctr, backoff=5, up_vector = (0,0,1), backoff_vector=(0,1,0))

    eval_actor_image([mesh_actor], 'full_cell_show_links.png', tmp_path, camera=camera)
              



def test_ngl_state(full_cell_mesh, tmp_path):
    with open('test/test_files/view_state.json', 'r') as fp:
        ngl_state = json.load(fp)
    
    camera = trimesh_vtk.camera_from_ngl_state(ngl_state)
    mesh_actor = trimesh_vtk.mesh_actor(full_cell_mesh)
    eval_actor_image([mesh_actor], 'full_cell_ngl_view.png', tmp_path, camera=camera)
    

def test_point_cloud(full_cell_mesh, full_cell_synapses, full_cell_soma_pt, tmp_path):

    mesh_actor = trimesh_vtk.mesh_actor(full_cell_mesh)
    camera = trimesh_vtk.oriented_camera(full_cell_soma_pt, backoff=300)
    sizes = full_cell_synapses['sizes']

    # size points by size, fixed color
    syn_actor = trimesh_vtk.point_cloud_actor(full_cell_synapses['positions'],
                                              size=sizes,
                                              color=(1,0,0))
    eval_actor_image([mesh_actor, syn_actor], 'full_cell_with_synapes_size_scaled.png', tmp_path, camera=camera)

    # color points by size, mapping sizes
    syn_actor = trimesh_vtk.point_cloud_actor(full_cell_synapses['positions'],
                                            size=500,
                                            color=np.clip(sizes, 0, 1000))
    eval_actor_image([mesh_actor, syn_actor], 'full_cell_synapes_colored_size.png', tmp_path, camera=camera)

    # color and size points
    syn_actor = trimesh_vtk.point_cloud_actor(full_cell_synapses['positions'],
                                            size=sizes,
                                            color=np.clip(sizes, 0, 1000))
    eval_actor_image([mesh_actor, syn_actor], 'full_cell_synapes_colored_and_size.png', tmp_path, camera=camera)

    # random colors
    x = np.linspace(0,1.0,len(sizes))   
    rand_colors = np.hstack([x[:, np.newaxis], 
                        np.abs(x-.5)[:, np.newaxis], 
                        (1-x)[:,np.newaxis]]) 
    
    syn_actor = trimesh_vtk.point_cloud_actor(full_cell_synapses['positions'],
                                            size=500,
                                            color=rand_colors)
    eval_actor_image([mesh_actor, syn_actor], 'full_cell_synapes_random_colors.png', tmp_path, camera=camera)

    # random colors uint8
    rand_colors_uint8 = np.uint8(rand_colors*255)
    syn_actor = trimesh_vtk.point_cloud_actor(full_cell_synapses['positions'],
                                              size=500,
                                              color=rand_colors_uint8)
    eval_actor_image([mesh_actor, syn_actor], 'full_cell_synapes_random_colors_uint8.png', tmp_path, camera=camera)

    # test failure modes
    with pytest.raises(ValueError) as e:
        syn_actor = trimesh_vtk.point_cloud_actor(full_cell_synapses['positions'],
                                                  size=np.random.rand(10,10),
                                                  color=(1,0,0))

    with pytest.raises(ValueError) as e:
        syn_actor = trimesh_vtk.point_cloud_actor(full_cell_synapses['positions'],
                                                  size=300,
                                                  color=np.random.rand(len(x),2))