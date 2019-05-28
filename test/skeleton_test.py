import numpy as np
from meshparty import skeleton
from skeleton_io_test import full_cell_skeleton, simple_skeleton, simple_verts, simple_edges
from scipy.sparse import csgraph
from copy import deepcopy

def test_basic_components(simple_skeleton):
    sk = simple_skeleton
    assert np.array_equal(sk.vertices, simple_verts)
    assert len(sk.edges) == len(simple_edges)
    assert sk.n_vertices == len(simple_verts)
    for sk_edge, base_edge in zip(sk.edges, simple_edges):
        assert np.all( np.sort(sk_edge) == np.sort(base_edge) )
        assert sk.distance_to_root[sk_edge[0]] > sk.distance_to_root[sk_edge[1]]

def test_reroot(simple_skeleton):
    sk = deepcopy(simple_skeleton)
    sk.reroot(6)
    assert sk.root == 6
    for sk_edge in sk.edges:
        assert sk.distance_to_root[sk_edge[0]] > sk.distance_to_root[sk_edge[1]]

def test_segments(simple_skeleton):
    sk = simple_skeleton
    assert len(sk.segments) == 3
    assert len(np.unique(np.concatenate(sk.segments))) == len(sk.vertices)
    assert np.all(sk.segment_map == np.array([1, 1, 1, 2, 2, 0, 0]))

def test_sk_csgraph(simple_skeleton):
    sk = simple_skeleton
    graph = sk.csgraph
    gdist = csgraph.dijkstra(graph, indices=[6])
    assert np.all(gdist[0] == np.array([ 4.,  3.,  2., np.inf, np.inf,  1.,  0.]))

    ugdist = csgraph.dijkstra(sk.csgraph_undirected, indices=[6])
    assert np.all(ugdist[0] == np.array([4., 3., 2., 3., 4., 1., 0.]))

    bg = sk.csgraph_binary.toarray()
    assert np.all(np.unique(bg) == [0,1])
    assert np.array_equal(bg, (sk.csgraph>0).toarray())