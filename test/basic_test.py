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
def mesh_with_extra_edges():

    verts = np.array([[0, 0, 0],
                      [1, 0, 0], 
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2],
                      [2, 3, 1],
                      [3, 4, 2]], np.uint32)
    ext_edges = np.array([[1,4],
                          [2,4]])
    mesh = trimesh_io.Mesh(verts, faces, mesh_edges=ext_edges, process=False)
    assert np.all(mesh.vertices == verts)
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


@pytest.fixture(scope='session')
def basic_mesh_id(cv, cv_folder, basic_mesh):
    mesh_id = 100
    mesh_dir = os.path.join(cv_folder, )
    if not os.path.isdir(mesh_dir):
        os.makedirs(mesh_dir)
    n_vertices = basic_mesh.vertices.shape[0]
    outs=struct.pack("=I", n_vertices) + basic_mesh.vertices.tobytes() + basic_mesh.faces.tobytes()
    with cloudvolume.Storage(cv.layer_cloudpath, progress=cv.progress) as stor:
        fname_man = os.path.join(cv.info['mesh'], f'{mesh_id}:0')
        frag_id = f'9{mesh_id}:0'
        fname = os.path.join(cv.info['mesh'], frag_id)
        d_man = {'fragments': [frag_id]}
        stor.put_json(fname_man, json.dumps(d_man))
        stor.put_file(fname, outs)
        print(fname, fname_man)
        print(np.max(basic_mesh.faces))
    yield mesh_id


def test_write_mesh(basic_mesh, tmpdir):

    filepath = str(tmpdir.join('test.h5'))
    trimesh_io.write_mesh_h5(filepath, basic_mesh.vertices, basic_mesh.faces)

    new_mesh = trimesh_io.read_mesh_h5(filepath)
    assert(np.all(basic_mesh.vertices == new_mesh[0]))


def test_meta_mesh(cv_path, basic_mesh_id):
    mm = trimesh_io.MeshMeta(cv_path=cv_path)
    mesh = mm.mesh(seg_id=basic_mesh_id,
                   merge_large_components=False,
                   remove_duplicate_vertices=False)
    assert(mesh is not None)
