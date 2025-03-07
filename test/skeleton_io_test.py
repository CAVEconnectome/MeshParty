import tempfile
import numpy as np
import pandas as pd
import pytest
import itertools
import contextlib
import os
from meshparty import skeleton_io, skeleton

overwrite_flags = [True, False]
io_file_exists = [True, False]

simple_verts = np.array(
    [[1, 0, 5], [1, 1, 5], [1, 2, 5], [0, 2, 5], [0, 2, 4], [2, 2, 5], [2, 2, 6]],
    dtype=np.int32,
)

simple_edges = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6]], dtype=np.int32
)


@contextlib.contextmanager
def build_full_cell_skeleton():
    filepath = "test/test_files/sk_648518346349499581.h5"
    sk = skeleton_io.read_skeleton_h5(filepath, remove_zero_length_edges=False)
    sk._rooted._mesh_index = np.array(sk.vertex_properties["mesh_index"])
    yield sk


@pytest.fixture(scope="session")
def full_cell_skeleton():
    with build_full_cell_skeleton() as sk:
        yield sk


@pytest.fixture(scope="session")
def simple_skeleton():
    verts = simple_verts
    edges = simple_edges
    yield skeleton.Skeleton(verts, edges, root=0)


@pytest.fixture(scope="session")
def simple_skeleton_with_properties():
    verts = simple_verts
    edges = simple_edges
    mesh_index = np.arange(0, 10 * len(verts), 10)
    mesh_to_skel_map = np.repeat(np.arange(0, len(verts)), 3)
    test_prop = mesh_index.copy()
    yield skeleton.Skeleton(
        verts,
        edges,
        mesh_index=mesh_index,
        mesh_to_skel_map=mesh_to_skel_map,
        vertex_properties={"test": test_prop},
        root=0,
    )


@pytest.mark.parametrize(
    "overwrite_flag,file_exist", itertools.product(overwrite_flags, io_file_exists)
)
def test_skeleton_read_write(simple_skeleton, overwrite_flag, file_exist):
    sk = simple_skeleton

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=(not file_exist)) as tf:
        fname = tf.name

    if file_exist and not overwrite_flag:
        with pytest.raises(FileExistsError):
            skeleton_io.write_skeleton_h5(sk, fname, overwrite=overwrite_flag)
    else:
        skeleton_io.write_skeleton_h5(sk, fname, overwrite=overwrite_flag)

    if file_exist and not overwrite_flag:
        with pytest.raises(OSError):
            skback = skeleton_io.read_skeleton_h5(fname)
    else:
        skback = skeleton_io.read_skeleton_h5(fname)
        assert np.all(skback.vertices == sk.vertices)
        assert np.all(skback.edges == sk.edges)
        assert skback.root == sk.root


def test_skeleton_read_write_with_props(simple_skeleton_with_properties, tmp_path):
    sk = simple_skeleton_with_properties

    fname = os.path.join(tmp_path, "test.swc")
    skeleton_io.write_skeleton_h5(sk, fname, overwrite=True)
    skback = skeleton_io.read_skeleton_h5(fname)
    assert np.all(skback.vertex_properties["test"] == sk.vertex_properties["test"])
    assert np.all(skback.mesh_to_skel_map == sk.mesh_to_skel_map)


def test_swc_write(simple_skeleton_with_properties, tmp_path):
    sk = simple_skeleton_with_properties
    fname = os.path.join(tmp_path, "test.swc")

    labels = [0, 1, 1, 1, 3, 3, 3]
    skeleton_io.export_to_swc(
        sk,
        fname,
        radius=sk.vertex_properties["test"],
        node_labels=labels,
        xyz_scaling=1,
    )
    sk_pd = pd.read_csv(
        fname,
        sep=" ",
        header=None,
        names=["index", "type", "x", "y", "z", "r", "parent"],
    )
    assert sk_pd.loc[3].parent == 2

    sk.export_to_swc(
        fname, radius=sk.vertex_properties["test"], node_labels=labels, xyz_scaling=1
    )
    sk_pd = pd.read_csv(
        fname,
        sep=" ",
        header=None,
        names=["index", "type", "x", "y", "z", "r", "parent"],
    )
    assert sk_pd.loc[3].parent == 2


def test_skeleton_h5_read(full_cell_skeleton):
    print(full_cell_skeleton.root)
    assert type(full_cell_skeleton) is skeleton.Skeleton
    assert full_cell_skeleton.root == 42197


def test_skeleton_rescale(full_cell_skeleton):
    end_points = full_cell_skeleton.end_points
    orig_dist = full_cell_skeleton.distance_to_root[end_points[10]]

    full_cell_skeleton.voxel_scaling = [2, 2, 1]
    new_dist = full_cell_skeleton.distance_to_root[end_points[10]]
    assert not np.isclose(new_dist, orig_dist)
    assert np.isclose(new_dist, 249974.84, atol=0.01)

    full_cell_skeleton.voxel_scaling = None
    reset_dist = full_cell_skeleton.distance_to_root[end_points[10]]
    assert np.isclose(orig_dist, reset_dist)
