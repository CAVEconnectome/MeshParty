import numpy as np
from meshparty import skeleton
from skeleton_io_test import full_cell_skeleton, simple_skeleton, simple_verts, simple_edges

def test_basic_components(simple_skeleton):
    sk = simple_skeleton
    assert np.array_equal(sk.vertices, simple_verts)
    assert len(sk.edges) == len(simple_edges)
    assert sk.n_vertices == len(simple_verts)
    for sk_edge, base_edge in zip(sk.edges, simple_edges):
        assert np.all( np.sort(sk_edge) == np.sort(base_edge) )
        assert sk.distance_to_root[sk_edge[0]] > sk.distance_to_root[sk_edge[1]]



def test_reroot(simple_skeleton):
    sk = simple_skeleton
    sk.reroot(6)
    assert sk.root == 6
    for sk_edge in sk.edges:
        assert sk.distance_to_root[sk_edge[0]] > sk.distance_to_root[sk_edge[1]]
