from meshparty import trimesh_io, skeletonize, mesh_filters, skeleton
import numpy as np
import pytest
import cloudvolume
import json
import os
import struct
import contextlib
import json


@contextlib.contextmanager
def build_basic_mesh():
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
def basic_mesh():
    with build_basic_mesh() as m:
        yield m


@contextlib.contextmanager
def build_full_cell_mesh():
    filepath = 'test/test_files/648518346349499581.h5'
    vertices, faces, normals, link_edges, node_mask = trimesh_io.read_mesh_h5(
        filepath)
    mesh = trimesh_io.Mesh(vertices, faces)
    yield mesh


@contextlib.contextmanager
def build_full_cell_merge_log():
    filepath = 'test/test_files/648518346349499581_merge_log.json'
    with open(filepath, 'r') as fp:
        merge_log = json.load(fp)
    yield merge_log


@contextlib.contextmanager
def build_full_cell_synapses():
    filepath = 'test/test_files/648518346349499581_synapses.json'
    with open(filepath, 'r') as fp:
        synapse_d = json.load(fp)
    yield synapse_d


@pytest.fixture(scope='module')
def full_cell_soma_pt():
    return np.array([358304, 219012,  53120])


@pytest.fixture(scope='session')
def full_cell_merge_log():
    with build_full_cell_merge_log() as ml:
        yield ml


@pytest.fixture(scope='session')
def full_cell_synapses():
    with build_full_cell_synapses() as ml:
        ml['positions'] = np.array(ml['positions'])
        ml['sizes'] = np.array(ml['sizes'])
        yield ml


@pytest.fixture(scope='session')
def full_cell_mesh():
    with build_full_cell_mesh() as m:
        yield m


@pytest.fixture(scope='session')
def mesh_link_edges():
    filepath = 'test/test_files/link_edges_for_mesh.npy'
    link_edges = np.load(filepath)
    yield link_edges


@contextlib.contextmanager
def build_basic_cube_mesh():
    verts = np.array([
        [-1., -1.,  1.],
        [-1., -1., -1.],
        [1., -1., -1.],
        [1., -1.,  1.],
        [-1.,  1.,  1.],
        [-1.,  1., -1.],
        [1.,  1., -1.],
        [1.,  1.,  1.]], dtype=np.float32)
    faces = np.array([
        [4, 5, 1],
        [5, 6, 2],
        [6, 7, 3],
        [7, 4, 0],
        [0, 1, 2],
        [7, 6, 5],
        [0, 4, 1],
        [1, 5, 2],
        [2, 6, 3],
        [3, 7, 0],
        [3, 0, 2],
        [4, 7, 5]], np.uint32)
    mesh = trimesh_io.Mesh(verts, faces, process=False)
    assert np.all(mesh.vertices == verts)
    yield mesh


@pytest.fixture(scope='session')
def basic_cube_mesh():
    with build_basic_cube_mesh() as m:
        yield m


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
    cv = cloudvolume.CloudVolume(cloudpath=cv_path,
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
        np.uint32(n_vertices),  # Number of vertices (3 coordinates)
        vertices,
        np.array(mesh.faces, dtype=np.uint32)
    ]
    outs = b''.join([array.tobytes() for array in vertex_index_format])

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


def test_get_mesh(cv_path, basic_mesh_id, tmpdir):
    mm = trimesh_io.MeshMeta(cv_path=cv_path, disk_cache_path=tmpdir)
    basic_mesh = mm.mesh(seg_id=basic_mesh_id)
    assert(basic_mesh.n_vertices == 5)


def test_download_meshes(cv_path, basic_mesh_id, full_cell_mesh_id, tmpdir):
    meshes = trimesh_io.download_meshes([basic_mesh_id, full_cell_mesh_id],
                                        tmpdir,
                                        cv_path=cv_path,
                                        merge_large_components=False)
    meshes = trimesh_io.download_meshes([basic_mesh_id, full_cell_mesh_id],
                                        tmpdir,
                                        cv_path=cv_path,
                                        merge_large_components=False,
                                        n_threads=2)


def test_write_mesh(basic_mesh, tmpdir):

    filepath = str(tmpdir.join('test.h5'))
    basic_mesh.write_to_file(filepath)

    new_mesh = trimesh_io.read_mesh_h5(filepath)
    assert(np.all(basic_mesh.vertices == new_mesh[0]))


def test_meta_mesh(cv_path, basic_mesh_id, full_cell_mesh_id, tmpdir):
    mm = trimesh_io.MeshMeta(cv_path=cv_path)
    mesh = mm.mesh(seg_id=basic_mesh_id)
    full_cell_mesh = mm.mesh(seg_id=full_cell_mesh_id,
                             merge_large_components=False)
    assert(mesh is not None)
    assert(full_cell_mesh is not None)


def test_masked_mesh(cv_path, full_cell_mesh_id, full_cell_soma_pt, tmpdir):
    mm = trimesh_io.MeshMeta(cv_path=cv_path,
                             cache_size=0,
                             disk_cache_path=os.path.join(tmpdir, 'mesh_cache'))
    mmesh = mm.mesh(seg_id=full_cell_mesh_id)

    assert(mmesh is not None)
    # read again to test file caching with memory caching on

    mmesh_cache = mm.mesh(seg_id=full_cell_mesh_id)

    # now set it up with memory caching enabled
    mm = trimesh_io.MeshMeta(cv_path=cv_path,
                             cache_size=1)
    # read it again with memory caching enabled
    mmesh = mm.mesh(seg_id=full_cell_mesh_id)
    # read it again to use memory cache
    mmesh_mem_cache = mm.mesh(seg_id=full_cell_mesh_id)

    ds = np.linalg.norm(mmesh.vertices - full_cell_soma_pt, axis=1)
    soma_mesh = mmesh.apply_mask(ds < 15000)

    is_big = mesh_filters.filter_largest_component(soma_mesh)
    soma_mesh = soma_mesh.apply_mask(is_big)

    ds = np.linalg.norm(soma_mesh.vertices - full_cell_soma_pt, axis=1)
    double_soma_mesh = soma_mesh.apply_mask(ds < 10000)

    with pytest.raises(ValueError):
        bad_mask = mmesh.apply_mask([True, True])

    random_indices = np.array([0, 500, 1500])
    orig_indices = double_soma_mesh.map_indices_to_unmasked(random_indices)
    back_indices = double_soma_mesh.filter_unmasked_indices(orig_indices)

    assert np.all(random_indices == back_indices)

    fname = os.path.join(tmpdir, 'test_mask_mesh.h5')
    double_soma_mesh.write_to_file(fname)

    double_soma_read = mm.mesh(filename=fname)


def test_link_edges(full_cell_mesh, full_cell_merge_log, full_cell_soma_pt):

    lcc_before = mesh_filters.filter_largest_component(full_cell_mesh)
    assert lcc_before.sum() == 1973174

    full_cell_mesh.voxel_scaling = [10, 10, 10]
    full_cell_mesh.add_link_edges(
        merge_log=full_cell_merge_log, base_resolution=[1, 1, 1])
    full_cell_mesh.voxel_scaling = None

    lcc_after = mesh_filters.filter_largest_component(full_cell_mesh)
    assert lcc_after.sum() == 2188351


def test_local_mesh(full_cell_mesh):
    vertex = 30000
    local_mesh = full_cell_mesh.get_local_mesh(
        n_points=500, max_dist=5000, center_node_id=vertex)
    assert(len(local_mesh.vertices) == 500)
    local_view = full_cell_mesh.get_local_view(
        n_points=500, max_dist=5000, center_node_id=vertex)
    assert(len(local_view[0][0]) == 500)


def test_mesh_rescale(full_cell_mesh):
    original_area = full_cell_mesh.area

    full_cell_mesh.voxel_scaling = [2, 2, 1]
    new_area = full_cell_mesh.area
    assert np.isclose(51309567968.120735, new_area, atol=1)

    full_cell_mesh.voxel_scaling = None
    restored_area = full_cell_mesh.area
    assert original_area == restored_area


def test_mesh_meta_rescale(cv_path, full_cell_mesh_id, tmpdir):
    mm = trimesh_io.MeshMeta(cv_path=cv_path,
                             cache_size=0,
                             disk_cache_path=os.path.join(
                                 tmpdir, 'mesh_cache'),
                             voxel_scaling=[2, 2, 1])
    mmesh = mm.mesh(seg_id=full_cell_mesh_id)
    assert np.isclose(51309567968.120735, mmesh.area, atol=1)
