import numpy as np
from meshparty import skeleton, skeleton_quality
import pytest
from .skeleton_io_test import full_cell_skeleton, simple_skeleton, simple_skeleton_with_properties, simple_verts, simple_edges
from .basic_test import full_cell_mesh, mesh_link_edges
from scipy.sparse import csgraph
from copy import deepcopy


def test_basic_components(simple_skeleton, simple_skeleton_with_properties):
    sk = deepcopy(simple_skeleton)
    assert np.array_equal(sk.vertices, simple_verts)
    assert len(sk.edges) == len(simple_edges)
    assert sk.n_vertices == len(simple_verts)
    for sk_edge, base_edge in zip(sk.edges, simple_edges):
        assert np.all(np.sort(sk_edge) == np.sort(base_edge))
        assert sk.distance_to_root[sk_edge[0]
                                   ] > sk.distance_to_root[sk_edge[1]]

    d, _ = sk.kdtree.query([0.3, 0.4, 0.1])
    assert np.isclose(d, 4.2261093, atol=0.0001)

    d, _ = sk.pykdtree.query(np.array([[0.3, 0.4, 0.1]]))
    assert np.isclose(d, 4.2261093, atol=0.0001)

    skp = simple_skeleton_with_properties
    assert np.all(skp.vertex_properties['test'] == skp.mesh_index)


def test_skeleton_creation(simple_skeleton):
    sk = simple_skeleton
    new_sk = skeleton.Skeleton(sk.vertices,
                               sk.edges, root=None)
    assert new_sk.root is not None

    new_sk = skeleton.Skeleton(sk.vertices,
                               sk.edges,
                               root=3)
    assert new_sk.root == 3

    with pytest.raises(ValueError):
        skeleton.Skeleton(sk.vertices,
                          sk.edges,
                          root=len(sk.vertices)+1)


def test_segments(simple_skeleton):
    sk = skeleton.Skeleton(simple_skeleton.vertices,
                           simple_skeleton.edges, root=0)
    assert len(sk.segments) == 3
    assert len(np.unique(np.concatenate(sk.segments))) == len(sk.vertices)
    assert np.all(sk.segment_map == np.array([0, 0, 0, 1, 1, 2, 2]))


def test_reroot(simple_skeleton):
    sk = deepcopy(simple_skeleton)
    sk.reroot(6)
    assert sk.root == 6
    for sk_edge in sk.edges:
        assert sk.distance_to_root[sk_edge[0]
                                   ] > sk.distance_to_root[sk_edge[1]]


def test_sk_csgraph(simple_skeleton):
    sk = simple_skeleton
    graph = sk.csgraph
    gdist = csgraph.dijkstra(graph, indices=[6])
    assert np.all(gdist[0] == np.array(
        [4.,  3.,  2., np.inf, np.inf,  1.,  0.]))

    ugdist = csgraph.dijkstra(sk.csgraph_undirected, indices=[6])
    assert np.all(ugdist[0] == np.array([4., 3., 2., 3., 4., 1., 0.]))

    bg = sk.csgraph_binary.toarray()
    assert np.all(np.unique(bg) == [0, 1])
    assert np.array_equal(bg, (sk.csgraph > 0).toarray())

    ubg = sk.csgraph_binary_undirected.toarray()
    assert np.all(np.unique(ubg) == [0, 1])
    assert np.array_equal(ubg, (sk.csgraph_undirected > 0).toarray())


def test_branch_and_endpoints(full_cell_skeleton):
    sk = full_cell_skeleton

    assert len(sk.end_points) == 73
    assert sk.n_end_points == 73
    assert len(sk.branch_points) == 66
    assert sk.n_branch_points == 66

    path = sk.path_to_root(sk.end_points[1])
    assert np.isclose(sk.path_length(path), 156869.06, atol=0.01)


def test_cover_paths(full_cell_skeleton):
    sk = full_cell_skeleton
    cover_paths = sk.cover_paths
    assert len(np.unique(np.concatenate(cover_paths))) == len(sk.vertices)
    assert cover_paths[0][-1] == sk.root
    assert len(cover_paths) == sk.n_end_points


def test_cut_graph(full_cell_skeleton):
    sk = full_cell_skeleton
    nc, _ = csgraph.connected_components(sk.csgraph)
    assert nc == 1
    assert sk.csgraph[300].nnz == 1

    cg = sk.cut_graph(300)
    nc, _ = csgraph.connected_components(cg)
    assert nc == 2
    assert cg[300].nnz == 0


def test_child_nodes(full_cell_skeleton):
    sk = full_cell_skeleton
    assert len(sk.child_nodes(sk.branch_points[0])) == 2
    assert len(np.concatenate(sk.child_nodes(sk.end_points))) == 0


def test_downstream_nodes(full_cell_skeleton):
    sk = full_cell_skeleton
    assert len(sk.downstream_nodes(sk.root)) == sk.n_vertices
    assert len(sk.downstream_nodes(300)) == 135


def test_skeleton_quality(full_cell_skeleton, full_cell_mesh, mesh_link_edges):
    sk = full_cell_skeleton
    mesh = full_cell_mesh
    mesh.link_edges = mesh_link_edges
    pscore, sk_paths, ms_paths, sk_inds_list, mesh_inds_list, path_distances = \
        skeleton_quality.skeleton_path_quality(sk, mesh, return_path_info=True)
    assert len(pscore) == len(sk.cover_paths)
    assert np.isclose(pscore.sum(), -151.377, 0.001)
