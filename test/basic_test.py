from meshparty import trimesh_io
import numpy as np
import pytest
import cloudvolume
import json
import os
import struct


@pytest.fixture(scope='session')
def basic_mesh():

    verts = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2],
                      [2, 3, 1],
                      [3, 4, 2]], np.uint32)
    mesh = trimesh_io.Mesh(verts, faces, process=False)
    assert np.all(mesh.vertices == verts)
    yield mesh


@pytest.fixture(scope='session')
def full_cell_mesh():
    filepath = 'test/test_files/648518346349499581.h5'
    vertices, faces, normals, mesh_edges = trimesh_io.read_mesh_h5(filepath)
    mesh = trimesh_io.Mesh(vertices, faces, process=False)
    yield mesh


@pytest.fixture(scope='session')
def cv_folder(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp('test_cv'))
    yield tmpdir


@pytest.fixture(scope='session')
def cv_path(cv_folder):
    cv_path = "precomputed://file://"+str(cv_folder)
    yield cv_path


@pytest.fixture(scope='session')
def cv(cv_path):

    info = cloudvolume.CloudVolume.create_new_info(
        num_channels=1,
        layer_type='segmentation',
        data_type='uint64',
        encoding='raw',
        resolution=[4, 4, 40],
        voxel_offset=[0, 0, 0],
        mesh='mesh',
        chunk_size=[512, 512, 16],
        volume_size=[512, 512, 512]
    )
    cv = cloudvolume.CloudVolumeFactory(cloudurl=cv_path,
                                        info=info)
    cv.commit_info()

    yield cv

def write_mesh_to_cv(cv, cv_folder, mesh, mesh_id):
    mesh_dir = os.path.join(cv_folder, )
    if not os.path.isdir(mesh_dir):
        os.makedirs(mesh_dir)
    n_vertices = mesh.vertices.shape[0]
    
    vertices = np.array(mesh.vertices, dtype=np.float32)

    vertex_index_format = [
        np.uint32(n_vertices), # Number of vertices (3 coordinates)
        vertices,
        np.array(mesh.faces, dtype=np.uint32)
    ]
    outs= b''.join([array.tobytes() for array in vertex_index_format])

    with cloudvolume.Storage(cv.layer_cloudpath, progress=cv.progress) as stor:
        fname_man = os.path.join(cv.info['mesh'], f'{mesh_id}:0')
        frag_id = f'9{mesh_id}:0'
        fname = os.path.join(cv.info['mesh'], frag_id)
        d_man = {'fragments': [frag_id]}
        stor.put_json(fname_man, json.dumps(d_man))
        stor.put_file(
            file_path=fname,
            content=outs,
            compress=True
        )


@pytest.fixture(scope='session')
def basic_mesh_id(cv, cv_folder, basic_mesh):
    mesh_id = 100
    write_mesh_to_cv(cv, cv_folder, basic_mesh, mesh_id)
    yield mesh_id


@pytest.fixture(scope='session')
def full_cell_mesh_id(cv, cv_folder, full_cell_mesh):
    mesh_id = 101
    write_mesh_to_cv(cv, cv_folder, full_cell_mesh, mesh_id)
    yield mesh_id


def test_write_mesh(basic_mesh, tmpdir):

    filepath = str(tmpdir.join('test.h5'))
    trimesh_io.write_mesh_h5(filepath, basic_mesh.vertices, basic_mesh.faces)

    new_mesh = trimesh_io.read_mesh_h5(filepath)
    assert(np.all(basic_mesh.vertices == new_mesh[0]))


def test_meta_mesh(cv_path, basic_mesh_id, full_cell_mesh_id):
    mm = trimesh_io.MeshMeta(cv_path=cv_path)
    mesh = mm.mesh(seg_id=basic_mesh_id)
    full_cell_mesh = mm.mesh(seg_id=full_cell_mesh_id,
                             merge_large_components=False,
                             remove_duplicate_vertices=False)
    assert(mesh is not None)
    #assert(full_cell_mesh is not None)
