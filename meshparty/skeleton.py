import numpy as np
from meshparty import utils
from scipy import spatial, sparse
from pykdtree.kdtree import KDTree
from copy import copy
import json


def load_from_json(path, use_smooth_vertices=False):
    with open(path, 'r') as fp:
        d = json.load(fp)

    if use_smooth_vertices:
        assert 'smooth_vertices' in d
        skel_vertices = np.array(d['smooth_vertices'], dtype=np.float)
    else:
        skel_vertices = np.array(d['vertices'], dtype=np.float)

    if "root" in d:
        root = np.array(d['root'], dtype=np.float)
    else:
        root = None

    skel_edges = np.array(d['edges'], dtype=np.int64)

    return SkeletonForest(skel_vertices, skel_edges, root=root)


class SkeletonForest:
    def __init__(self, vertices, edges, vertex_properties={},
                 edge_properties={}, vertex_lists={}, root=None):
        self._vertex_components = np.full(len(vertices), None)
        self._vertex_order = np.full(len(vertices), 0, dtype=int)  # Original order of vertices
        self._edge_components = np.full(len(edges), None)
        self._vertex_lists = {}
        self._skeletons = []
        self._kdtree = None
        self._csgraph = None
        self._csgraph_binary = None
        self._root = None

        vertices = np.array(vertices)
        edges = np.array(edges)
        bin_csgraph = utils.create_csgraph(vertices, edges,
                                           euclidean_weight=False)

        nc, v_lbls = sparse.csgraph.connected_components(bin_csgraph)
        lbls, count = np.unique(v_lbls, return_counts=True)
        lbl_order = np.argsort(count)[::-1]

        ind_base = 0
        for lbl in lbls[lbl_order]:
            v_filter = np.flatnonzero(v_lbls == lbl)
            vertices_f, edges_f, filters = utils.reduce_vertices(vertices,
                                                                 edges,
                                                                 v_filter=v_filter,
                                                                 return_filter_inds=True)
            vertex_properties_f = {vp_n: np.array(vp_v)[filters[0]]
                                   for vp_n, vp_v in vertex_properties.items()}
            edge_properties_f = {ep_n: np.array(ep_v)[filters[1]]
                                 for ep_n, ep_v in edge_properties.items()}

            self._vertex_components[filters[0]] = lbl
            self._edge_components[filters[1]] = lbl
            if root in v_filter:
                root_f = np.where(root == v_filter)[0][0]
                self._root = ind_base + root_f
            else:
                root_f = None
            self._vertex_order[v_filter] = ind_base + np.arange(len(vertices_f))
            ind_base += len(vertices_f)
            self._skeletons.append(Skeleton(vertices_f, edges_f,
                                            vertex_properties_f,
                                            edge_properties_f,
                                            root=root_f))

        for list_name, indices in vertex_lists.items():
            self.add_vertex_list(list_name, indices, remap_from_original_order=True)


    def __getitem__(self, key):
        return self._skeletons[key]

    @property
    def vertices(self):
        vs_list = [skeleton.vertices for skeleton in self._skeletons]
        return np.vstack(vs_list)
    
    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def edges(self):
        return self._agglomerate_nodes_across_skeletons('edges')

    @property
    def end_points(self):
        return self._agglomerate_nodes_across_skeletons('end_points')

    @property
    def branch_points(self):
        return self._agglomerate_nodes_across_skeletons('branch_points')

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._kdtree = KDTree(self.vertices)
        return self._kdtree

    @property
    def csgraph(self):
        if self._csgraph is None:
            self._csgraph = self._create_csgraph()
        return self._csgraph.copy()

    @property
    def csgraph_binary(self):
        if self._csgraph_binary is None:
            self._csgraph_binary = self._create_csgraph(euclidean_weight=False)
        return self._csgraph_binary

    @property
    def vertex_properties(self):
        vp = {}
        for vp_name in self._skeletons[0].vertex_properties.keys(): 
            vp[vp_name] = self._vertex_property(vp_name)
        return vp

    @property
    def edge_properties(self):
        ep = {}
        for ep_name in self._skeletons[0].edge_properties.keys(): 
            ep[ep_name] = self._edge_property(ep_name)
        return ep


    @property
    def root(self):
        return self._root
    

    def _vertex_property(self, property_name):
        vp_list = [skeleton.vertex_properties[property_name]
                   for skeleton in self._skeletons if len(skeleton.vertices) > 1]
        return np.concatenate(vp_list)

    def _edge_property(self, property_name):
        ep_list = [skeleton.edge_properties[property_name]
                   for skeleton in self._skeletons]
        return np.concatenate(ep_list)

    def _create_csgraph(self, euclidean_weight=True, directed=False):
        return utils.create_csgraph(self.vertices, self.edges,
                                    euclidean_weight=euclidean_weight,
                                    directed=directed)

    def _agglomerate_nodes_across_skeletons(self, property_name):
        nodes_stacked = []
        n_shift = 0
        for skeleton in self._skeletons:
            node_inds = getattr(skeleton, property_name)
            if len(node_inds) > 0:
                nodes_stacked.append(node_inds + n_shift)
                n_shift += skeleton.n_vertices
        if len(nodes_stacked) > 0:
            return np.concatenate(nodes_stacked)
        else:
            return np.array([])

    def add_vertex_list(self, name, vertex_list, remap_from_original_order=False):
        if remap_from_original_order:
            self._vertex_lists[name] = self.remap_vertex_list(vertex_list)
        else:
            self._vertex_lists[name] = vertex_list


    @property
    def vertex_lists(self):
        return self._vertex_lists

    def remap_vertex_list(self, vertex_list):
        '''
        Given a vertex list from the original coordinates, returns the version
        for the internally managed vertex list
        '''
        new_vertex_list = np.full(len(vertex_list), np.nan)
        new_vertex_list[~np.isnan(vertex_list)] = np.take(self._vertex_order,
                                                          vertex_list[~np.isnan(vertex_list)].astype(int))
        return new_vertex_list


class Skeleton:
    def __init__(self, vertices, edges, vertex_properties={},
                 edge_properties={}, root=None):
        self._vertices = np.array(vertices)
        self._edges = np.vstack(edges).astype(int)
        self.vertex_properties = vertex_properties
        self.edge_properties = edge_properties

        self._root = None
        self._paths = None
        self._parent_node_array = None
        self._kdtree = None
        self._csgraph = None
        self._csgraph_binary = None
        self._branch_points = None
        self._end_points = None

        self._reset_derived_objects()
        if root is None:
            self._create_default_root()
        else:
            self.reroot(root)

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def edges(self):
        return self._edges.copy()

    @property
    def csgraph(self):
        if self._csgraph is None:
            self._csgraph = self._create_csgraph()
        return self._csgraph.copy()

    @property
    def csgraph_binary(self):
        if self._csgraph_binary is None:
            self._csgraph_binary = self._create_csgraph(euclidean_weight=False)
        return self._csgraph_binary

    @property
    def csgraph_undirected(self):
        return self.csgraph + self.csgraph.T

    @property
    def csgraph_binary_undirected(self):
        return self.csgraph_binary + self.csgraph_binary.T

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def root(self):
        if self._root is None:
            self._create_default_root()
        return copy(self._root)

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._kdtree = KDTree(self.vertices)
        return self._kdtree

    @property
    def branch_points(self):
        if self._branch_points is None:
            self._create_branch_and_end_points()
        return self._branch_points.copy()

    @property
    def n_branch_points(self):
        if self._branch_points is None:
            self._create_branch_and_end_points()
        return len(self._branch_points)

    @property
    def end_points(self):
        if self._end_points is None:
            self._create_branch_and_end_points()
        return self._end_points.copy()

    @property
    def n_end_points(self):
        if self._end_points is None:
            self._create_branch_and_end_points()
        return len(self._end_points)

    @property
    def paths(self):
        if self._paths is None:
            self._paths = self._compute_paths()
        return self._paths

    @property
    def distance_to_root(self):
        ds = sparse.csgraph.dijkstra(self.csgraph, directed=False,
                                     indices=self.root)
        return ds

    def path_to_root(self, v_ind):
        '''
        Returns an ordered path to root from a given vertex node.
        '''
        path = [v_ind]
        ind = v_ind
        while ind is not None:
            ind = self._parent_node(ind)
            path.append(ind)
        return path

    def path_length(self, paths=None):
        if paths is None:
            paths = self.paths
        L = 0
        for path in paths:
            L += self._single_path_length(path)
        return L

    def reroot(self, new_root):
        self._root = new_root
        self._parent_node_array = np.full(self.n_vertices, None)

        # The edge list has to be treated like an undirected graph
        d = sparse.csgraph.dijkstra(self.csgraph_binary,
                                    directed=False,
                                    indices=new_root)

        # Make edges in edge list orient as [child, parent]
        # Where each child only has one parent
        # And the root has no parent. (Thus parent is closer than child)
        edges = self.edges
        is_ordered = d[edges[:, 0]] > d[edges[:, 1]]
        e1 = np.where(is_ordered, edges[:, 0], edges[:, 1])
        e2 = np.where(is_ordered, edges[:, 1], edges[:, 0])

        self._edges = np.stack((e1, e2)).T
        self._parent_node_array[e1] = e2
        self._reset_derived_objects()

    def _create_default_root(self):
        r = utils.find_far_points_graph(self._create_csgraph(directed=False))
        self.reroot(r[0])

    def _parent_node(self, vind):
        return self._parent_node_array[vind]

    def _reset_derived_objects(self):
        self._csgraph = None
        self._csgraph_binary = None
        self._paths = None

    def _create_csgraph(self,
                        directed=True,
                        euclidean_weight=True):

        return utils.create_csgraph(self.vertices, self.edges,
                                    euclidean_weight=euclidean_weight,
                                    directed=directed)

    def _create_branch_and_end_points(self):
        n_children = np.sum(self.csgraph_binary > 0, axis=0).squeeze()
        self._branch_points = np.flatnonzero(n_children > 1)
        self._end_points = np.flatnonzero(n_children == 0)

    def _single_path_length(self, path):
        xs = self.vertices[path[:-1]]
        ys = self.vertices[path[1:]]
        return sum(np.linalg.norm(ys-xs))

    def _compute_paths(self):
        '''
        Only considers the component with root
        '''
        ds, P = sparse.csgraph.dijkstra(self.csgraph,
                                        directed=True,
                                        indices=self.end_points,
                                        return_predecessors=True)
        d_to_root = ds[:, self.root]
        end_point_order = np.argsort(d_to_root)[::-1]
        paths = []

        visited = np.full(shape=(len(self.vertices),), fill_value=False)
        visited[self.root] = True
        for ep_ind in end_point_order:
            if np.isinf(d_to_root[ep_ind]):
                continue
            path, visited = self._unvisited_path_on_tree(self.end_points[ep_ind],
                                                         visited)
            paths.append(path)

        return paths

    def _unvisited_path_on_tree(self, ind, visited):
        '''
        Find path from ind to a visited node along G
        Assumes that G[i,j] means i->j
        '''
        n_ind = ind
        path = [n_ind]

        while not visited[n_ind]:
            visited[n_ind] = True
            n_ind = self._parent_node(n_ind)
            path.append(n_ind)
        return path, visited

    def _build_swc_array(self, node_labels, radius, xyz_scaling):
        '''
        Helper function for producing the numpy table for an swc.        
        '''
        ds = self.distance_to_root
        order_old = np.argsort(ds)
        new_ids = np.arange(len(ds))
        order_map = dict(zip(order_old, new_ids))

        node_labels = node_labels[order_old]
        xyz = self.vertices[order_old]
        radius = radius[order_old]
        par_ids = np.array([order_map.get(nid, -1) for nid in self._parent_node_array[order_old]])

        swc_dat = np.hstack((new_ids[:, np.newaxis],
                             node_labels[:, np.newaxis],
                             xyz / xyz_scaling,
                             radius[:, np.newaxis] / xyz_scaling,
                             par_ids[:, np.newaxis]))
        return swc_dat

    def export_to_swc(self, filename, node_labels=None, radius=None, header=None, xyz_scaling=1000):
        '''
        Export a skeleton file to an swc file
        (see http://research.mssm.edu/cnic/swc.html for swc definition)

        :param filename: Name of the file to save the swc to
        :param node_labels: None (default) or an interable of ints co-indexed with vertices.
                            Corresponds to the swc node categories. Defaults to setting all
                            nodes to label 3, dendrite.
        :param radius: None (default) or an iterable of floats. This should be co-indexed with vertices.
                       Radius values are assumed to be in the same units as the node vertices.
        :param header: Dict, default None. Each key value pair in the dict becomes
                       a parameter line in the swc header.
        :param xyz_scaling: Number, default 1000. Down-scales spatial units from the skeleton's units to
                            whatever is desired by the swc. E.g.

        '''

        if header is None:
            header_string = ''
        else:
            header_string = '\n'.join(['{}: {}'.format(k, v) for k, v in header.items()])

        if radius is None:
            radius = np.full(len(self.vertices), 1000)
        elif np.issubdtype(type(radius), int):
            radius = np.full(len(self.vertices), radius)

        if node_labels is None:
            node_labels = np.full(len(self.vertices), 3)

        swc_dat = self._build_swc_array(node_labels, radius, xyz_scaling)

        with open(filename, 'w') as f:
            np.savetxt(f, swc_dat, delimiter=' ', header=header_string, comments='#',
                       fmt=['%i', '%i', '%.3f', '%.3f', '%.3f', '%.3f', '%i'])
