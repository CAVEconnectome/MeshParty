from meshparty import trimesh_io
import numpy as np
import pytest


@pytest.fixture(scope='module')
def basic_mesh():

    verts = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1]])
    faces = np.array([[0, 1, 2],
                      [2, 3, 1],
                      [3, 4, 2]])
    mesh = trimesh_io.Mesh(verts, faces, process=False)
    assert np.all(mesh.vertices == verts)
    yield mesh


def test_write_mesh(basic_mesh, tmpdir):

    filepath = str(tmpdir.join('test.h5'))
    trimesh_io.write_mesh_h5(filepath, basic_mesh.vertices, basic_mesh.faces)

    new_mesh = trimesh_io.read_mesh_h5(filepath)
    assert(np.all(basic_mesh.vertices == new_mesh[0]))
