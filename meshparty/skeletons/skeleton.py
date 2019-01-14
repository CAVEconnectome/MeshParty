from scipy import sparse
import numpy as np
import time
from meshparty import trimesh_vtk
from meshparty import trimesh_io
import pandas as pd
from scipy.spatial import cKDTree as KDTree
from copy import copy
import pcst_fast  
from tqdm import trange, tqdm

from meshparty.skeletons import utils


class SkeletonForest():
    def __init__(self, vertices, edges, vertex_properties={}, edge_properties={}, root=None):
        self._vertex_components = np.full(len(vertices), None)
        self._edge_components = np.full(len(edges), None)
        self._skeletons = []
        self._kdtree = None

        vertices = np.array(vertices)
        edges = np.array(edges)

        nc, v_lbls = sparse.csgraph.connected_components(utils.create_csgraph(vertices, edges, euclidean_weight=False))
        lbls, count = np.unique(v_lbls, return_counts=True)
        lbl_order = np.argsort(count)[::-1]
        for lbl in lbls[lbl_order]:
            v_filter = np.where(v_lbls==lbl)[0]
            vertices_f, edges_f, filters = utils.reduce_vertices(vertices,
                                                           edges,
                                                           v_filter=v_filter,
                                                           return_filter_inds=True)
            vertex_properties_f = {vp_n: np.array(vp_v)[filters[0]] for vp_n, vp_v in vertex_properties.items()}
            edge_properties_f = {ep_n: np.array(ep_v)[filters[1]] for ep_n, ep_v in edge_properties.items()}

            self._vertex_components[filters[0]] = lbl
            self._edge_components[filters[1]] = lbl
            if root in v_filter:
                root_f = np.where(root==v_filter)[0][0]
            else:
                root_f = None
            self._skeletons.append( Skeleton(vertices_f, edges_f, vertex_properties_f, edge_properties_f, root=root_f) )

    def __getitem__(self, key):
        return self._skeletons[key]

    @property
    def vertices(self):
        vs_list = [skeleton.vertices for skeleton in self._skeletons]
        return np.vstack(vs_list)

    @property
    def edges(self):
        edges_stacked = []
        n_shift = 0
        for skeleton in self._skeletons:
            edges_stacked.append(skeleton.edges + n_shift)
            n_shift += skeleton.n_vertices
        return np.vstack(edges_stacked)

    @property
    def csgraph(self):    
        return utils.create_csgraph(self.vertices_all, self.edges_all)

    def vertex_property(self, property_name):
        vp_list = [skeleton.vertex_properties[property_name] for skeleton in self._skeletons if len(skeleton.vertices)>1]
        return np.concatenate(vp_list)

    def edge_property(self, property_name):
        ep_list = [skeleton.edge_properties[property_name] for skeleton in self._skeletons]
        return ep.concatenate(ep_list)

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._kdtree = KDTree(self.vertices)
        return self._kdtree


class Skeleton:
    def __init__(self, vertices, edges, vertex_properties={}, edge_properties={}, root=None):
        self._vertices = np.array(vertices)
        self._edges = np.vstack(edges).astype(int)
        self.vertex_properties = vertex_properties
        self.edge_properties = edge_properties

        self._reset_derived_objects()
        if root is None:
            self._create_default_root()
        else:
            self.reroot(root)

        # self._parent_node = {}
        self._kdtree = None
        self._branch_points = None
        self._end_points = None

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

    def _create_default_root(self):
        r = utils.find_far_points_graph(self._create_csgraph(directed=False))
        self.reroot(r[0])

    def _parent_node(self, vind):
        return self._parent_node_array[vind]

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
        is_ordered = d[edges[:,0]] > d[edges[:,1]]
        e1 = np.where( is_ordered, edges[:,0], edges[:,1])
        e2 = np.where( is_ordered, edges[:,1], edges[:,0])

        self._edges = np.stack((e1,e2)).T
        self._parent_node_array[e1]=e2
        self._reset_derived_objects()


    def _reset_derived_objects(self):
        self._csgraph = None
        self._csgraph_binary = None
        self._paths = None

    def _create_csgraph(self,
                        directed=True,
                        euclidean_weight=True):

        edges = self._edges.copy()

        if euclidean_weight:
            xs = self.vertices[edges[:,0]]
            ys = self.vertices[edges[:,1]]
            weights = np.linalg.norm(xs-ys, axis=1)
            use_dtype = np.float32
        else:   
            weights = np.ones((len(edges),)).astype(bool)
            use_dtype = bool 

        if directed:
            edges = edges.T
        else:
            edges = np.concatenate([edges.T, edges.T[[1, 0]]], axis=1)
            weights = np.concatenate([weights, weights]).astype(dtype=use_dtype)

        csgraph = sparse.csr_matrix((weights, edges),
                                    shape=[len(self.vertices), ] * 2,
                                    dtype=use_dtype)

        return csgraph


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

    def _create_branch_and_end_points(self):
        n_children = np.sum(self.csgraph_binary>0, axis=0).squeeze()
        self._branch_points = np.flatnonzero(n_children > 1)
        self._end_points = np.flatnonzero(n_children == 0)

    @property
    def paths(self):
        if self._paths is None:
            self._paths = self._compute_paths()
        return self._paths.copy()
    
    def distance_to_root(self, indices):
        ds = sparse.csgraph.dijkstra(self.csgraph, directed=True, indices=indices)
        return ds[:,self.root]

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

    def _single_path_length(self, path):
        xs = self.vertices[ path[:-1] ]
        ys = self.vertices[ path[1:] ]
        return sum(np.linalg.norm(ys-xs))

    def _compute_paths(self):
        '''
        Only considers the component with root
        '''
        ds, P = sparse.csgraph.dijkstra(self.csgraph,
                                        directed=True,
                                        indices=self.end_points,
                                        return_predecessors=True)
        d_to_root = ds[:,self.root]
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
        G = self.csgraph
        n_ind = ind
        path = [n_ind]
        while visited[n_ind] == False:
            visited[n_ind] = True
            n_ind = self._parent_node(n_ind)
            path.append(n_ind)
        return path, visited
