from meshparty import trimesh_io, trimesh_vtk, skeleton_io
import contextlib
import numpy as np
import pytest
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
    skel_actor = trimesh_vtk.skeleton_actor(cell_skel, vertex_property='rs', )
    pd = skel_actor.GetMapper().GetInput()
    verts_out, faces_out, edges_out = trimesh_vtk.poly_to_mesh_components(pd)
    assert(np.all(cell_skel.vertices == verts_out))
    assert(np.all(cell_skel.edges == edges_out))

    fname = 'test_sk_render.png'
    filepath = os.path.join(tmp_path, fname)
    trimesh_vtk.render_actors([skel_actor], do_save =True, filename= filepath, scale=1, back_color=(1,1,1))
    compare_img_to_test_file(filepath)


def test_full_cell_camera(full_cell_mesh, full_cell_soma_pt, tmp_path):
    mesh_actor = trimesh_vtk.mesh_actor(full_cell_mesh)
    camera = trimesh_vtk.oriented_camera(full_cell_soma_pt, backoff=100)
    fname = 'full_cell_orient_camera.png'
    filepath = os.path.join(tmp_path, fname)
    trimesh_vtk.render_actors([mesh_actor], do_save =True,
                                scale=1,
                                filename=filepath,
                                camera=camera, back_color=(1,1,1))
    compare_img_to_test_file(filepath)

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

    fname = 'full_cell_orient_camera.png'
    filepath = os.path.join(tmp_path, fname)
    trimesh_vtk.render_actors([mesh_actor], do_save =True,
                                scale=1,
                                filename=filepath,
                                back_color=(1,1,1))
    compare_img_to_test_file(filepath)

    mesh_actor = trimesh_vtk.mesh_actor(full_cell_mesh,
                                        opacity=1.0,
                                        show_link_edges = True)

    m1  =np.array(full_cell_merge_log['merge_edge_coords'][0]) 
    ctr = np.mean(m1, axis=0)
    camera = trimesh_vtk.oriented_camera(ctr, backoff=5, up_vector = (0,0,1), backoff_vector=(0,1,0))

  
    fname = 'full_cell_show_links.png'
    filepath = os.path.join(tmp_path, fname)
    trimesh_vtk.render_actors([mesh_actor], do_save =True,
                                scale=2,
                                camera=camera,
                                filename=filepath,
                                back_color=(1,1,1))
    compare_img_to_test_file(filepath)                  