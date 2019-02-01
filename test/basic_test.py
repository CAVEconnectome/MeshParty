from meshparty import trimesh_io
import numpy as np


def test_basic_mesh():

    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1]])
    faces = np.array([[0,1,2],
                      [2,3,1],
                      [3,4,2]])
    mesh = trimesh_io.Mesh(verts, faces, process=False)
    assert np.all(mesh.vertices == verts)
