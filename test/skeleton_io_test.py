import tempfile
import numpy as np
import pytest
import itertools
import contextlib

from meshparty import skeleton_io, skeleton

overwrite_flags = [True, False]
io_file_exists = [True, False]

@contextlib.contextmanager
def build_full_cell_skeleton():
    filepath = 'test/test_files/648518346349499581_sk.h5'
    sk = skeleton_io.read_skeleton_h5(filepath)
    yield sk

@pytest.fixture(scope='session')
def full_cell_skeleton():
    with build_full_cell_skeleton() as sk:
        yield sk

@pytest.fixture(scope='session')
def simple_skeleton():
    verts = np.array([[1, 0, 5],
                      [1, 1, 5],
                      [1, 2, 5],
                      [0, 2, 5],
                      [0, 2, 4],
                      [2, 2, 5],
                      [2, 2, 6]])

    edges = np.array([[0,1],
                      [1,2],
                      [2,3],
                      [3,4],
                      [2,5],
                      [5,6]])

    yield skeleton.Skeleton(verts, edges, root=0)

@pytest.mark.parametrize(
    'overwrite_flag,file_exist',
    itertools.product(overwrite_flags, io_file_exists))
def test_skeleton_read_write(simple_skeleton, overwrite_flag, file_exist):
    sk = simple_skeleton

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=(not file_exist)) as tf:
        fname = tf.name

    skeleton_io.write_skeleton_h5(sk, fname, overwrite=overwrite_flag)

    if file_exist and not overwrite_flag:
        with pytest.raises(OSError):
            skback = skeleton_io.read_skeleton_h5(fname)
    else:
        skback = skeleton_io.read_skeleton_h5(fname)
        assert np.all(skback.vertices==sk.vertices)
        assert np.all(skback.edges==sk.edges)
        assert skback.root == sk.root