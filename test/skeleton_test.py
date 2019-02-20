from meshparty import skeleton
import numpy as np
import pytest
import cloudvolume
import json
import os
import struct


@pytest.fixture(scope='session')
def basic_skeleton():

    vertices = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [2, 1, 0],
                      [2, 2, 2],
                      [2, 2, 0]], dtype=np.float)
    edges = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
		      [2, 4]], np.int64)

    root = np.array([[0,0,0]], dtype=np.float)
    skeleton = Skeleton(vertices, edges, root=root)
    assert np.all(skeleton.vertices == vertices)
    yield skeleton


@pytest.fixture(scope='session')
def skeleton_file():
    filename = "test/test_files/skeleton.json"
    yield filename

def test_root(basic_skeleton):
    numpy.testing.assert_array_equal(basic_skeleton.root, [0,0,0])


def test_n_vertices(basic_skeleton):
    assert (skeleton.n_vertices == 5)

def load_skeleton(basic_skeleton, skeleton_file):
    sk_forest = load_from_json(path, use_smooth_vertices=False)
    skel = sk_forest[0]
    assert np.all(skel.vertices == basic_skeleton.vertices)
    assert np.all(skel.edges == basic_skeleton.edges)
    assert np.all(skel.root == basic_skeleton.root)
