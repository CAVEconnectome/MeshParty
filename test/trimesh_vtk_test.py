from meshparty import trimesh_io, trimesh_vtk
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

@pytest.fixture(scope='session')
def cube_verts_faces():
    with build_basic_mesh() as m:
        yield m

def test_basic_mesh_actor(cube_verts_faces):
    verts, faces = cube_verts_faces
    mesh = trimesh_io.Mesh(verts, faces, process=False)
    mesh_actor = trimesh_vtk.make_mesh_actor(mesh)
    pd = mesh_actor.GetMapper().GetInput()
    verts_out, faces_out, tmp = trimesh_vtk.vtk_poly_to_mesh_components(pd)

    assert(np.all(verts == verts_out))
    assert(np.all(faces_out == faces))
