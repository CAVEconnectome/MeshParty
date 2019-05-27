from meshparty import trimesh_io, trimesh_vtk, skeleton_io
import contextlib
import numpy as np
import pytest

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
    mesh_actor = trimesh_vtk.make_mesh_actor(mesh)
    pd = mesh_actor.GetMapper().GetInput()
    verts_out, faces_out, tmp = trimesh_vtk.vtk_poly_to_mesh_components(pd)

    assert(np.all(verts == verts_out))
    assert(np.all(faces_out == faces))

def test_skeleton_viz(cell_skel):
    skel_actor = trimesh_vtk.vtk_skeleton_actor(cell_skel, vertex_property='rs')
    pd = skel_actor.GetMapper().GetInput()
    verts_out, faces_out, edges_out = trimesh_vtk.vtk_poly_to_mesh_components(pd)
    assert(np.all(cell_skel.vertices == verts_out))
    assert(np.all(cell_skel.edges == edges_out))
