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

def test_root(basic_skeleton):
    numpy.testing.assert_array_equal(basic_skeleton.root, [0,0,0])


def test_n_vertices(basic_skeleton):
    assert (skeleton.n_vertices == 5)

#def load_skeleton():
    
