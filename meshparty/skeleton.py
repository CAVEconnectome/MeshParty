import numpy as np
from meshparty import utils
from scipy import spatial, sparse
from pykdtree.kdtree import KDTree as pyKDTree
from copy import copy
import json
from meshparty import skeleton_io

class Skeleton:
    def __init__(self, vertices, edges, mesh_to_skel_map=None, vertex_properties={},
                 root=None):
        self._vertices = np.array(vertices)
        self._edges = np.vstack(edges).astype(int)
        self.vertex_properties = vertex_properties
        self._mesh_to_skel_map = mesh_to_skel_map

        self._root = None
        self._cover_paths = None
        self._segments = None
        self._segment_map = None

        self._parent_node_array = None
        self._kdtree = None
        self._pykdtree = None

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
        return self._vertices

    @property
    def edges(self):
        return self._edges

    @property
    def mesh_to_skel_map(self):
        return self._mesh_to_skel_map
    
    @property
    def segments(self):
        if self._segments is None:
            self._segments, self._segment_map = self._compute_segments()
        return self._segments

    @property
    def segment_map(self):
        if self._segment_map is None:
            self._segments, self._segment_map = self._compute_segments()
        return self._segment_map

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
    def pykdtree(self):
        if self._pykdtree is None:
            self._pykdtree = pyKDTree(self.vertices)
        return self._pykdtree

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._kdtree = spatial.cKDTree(self.vertices)
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
    def cover_paths(self):
        '''
        A list of paths from end nodes toward root that together fully cover the skeleton
        with no overlaps. Paths are ordered by end point distance from root, starting with
        the most distal ones.
        '''
        if self._cover_paths is None:
            self._cover_paths = self._compute_cover_paths()
        return self._cover_paths

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
        ind = self.parent_node(ind)
        while ind is not None:
            path.append(ind)
            ind = self.parent_node(ind)
        return path

    def path_length(self, paths=None):
        if paths is None:
            paths = self.paths
        L = 0
        for path in paths:
            L += self._single_path_length(path)
        return L

    def reroot(self, new_root):
        if new_root > self.n_vertices:
            raise ValueError('New root must be between 0 and n_vertices-1')
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


    def cut_graph(self, vinds, directed=True, euclidean_weight=True):
        '''
        Return a csgraph for the skeleton with each ind in inds cut off from its parent.
        '''
        e_keep = ~np.isin(self.edges[:,0], vinds)
        es_new = self.edges[e_keep]
        return utils.create_csgraph(self.vertices, es_new,
                                    euclidean_weight=euclidean_weight,
                                    directed=directed)


    def downstream_nodes(self, vinds):
        if np.isscalar(vinds):
            vinds = [vinds]
            return_single = True
        else:
            return_single = False

        dns = []
        for vind in vinds:
            g = self.cut_graph(vind)
            d = sparse.csgraph.dijkstra(g.T, indices=[vind])
            dns.append(np.flatnonzero(~np.isinf(d[0])))
        
        if return_single:
            dns=dns[0]
        return dns


    def child_nodes(self, vinds):
        if np.isscalar(vinds):
            vinds = [vinds]
            return_single = True
        else:
            return_single = False

        cinds = []
        for vind in vinds:
            cinds.append(self.edges[self.edges[:,1]==vind, 0])
        
        if return_single:
            cinds=cinds[0]
        return cinds

    def _create_default_root(self):
        r = utils.find_far_points_graph(self._create_csgraph(directed=False))
        self.reroot(r[0])

    def parent_node(self, vinds):
        return self._parent_node_array[vinds]

    def _reset_derived_objects(self):
        self._csgraph = None
        self._csgraph_binary = None
        self._cover_paths = None
        self._segments = None
        self._segment_map = None

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
        return np.sum(np.linalg.norm(ys-xs, axis=1))

    def _compute_cover_paths(self):
        '''
        Compute a list of cover paths along the skeleton
        '''
        cover_paths = []
        seen = np.full(self.n_vertices, False)
        ep_order = np.argsort(self.distance_to_root[self.end_points])[::-1]
        for ep in self.end_points[ep_order]:
            ptr = np.array(self.path_to_root(ep))
            cover_paths.append(ptr[~seen[ptr]])
            seen[ptr] = True
        return cover_paths

    def _compute_segments(self):
        segments = []
        segment_map = np.zeros(len(self.vertices))-1

        path_queue = self.end_points.tolist()
        bp_all = self.branch_points
        bp_seen = []
        seg_ind = 0

        while len(path_queue)>0:
            ind = path_queue.pop()
            segment = [ind]
            ptr = self.path_to_root(ind)
            if len(ptr)>1:
                for pind in ptr[1:]:
                    if pind in bp_all:
                        segments.append(np.array(segment))
                        segment_map[segment] = seg_ind
                        seg_ind += 1
                        if pind not in bp_seen:
                            path_queue.append(pind)
                            bp_seen.append(pind)
                        break
                    else:
                        segment.append(pind)
                else:
                    segments.append(np.array(segment))
                    segment_map[segment] = seg_ind
                    seg_ind += 1
            else:
                segments.append(np.array(segment))
                segment_map[segment] = seg_ind
                seg_ind += 1
        return segments, segment_map.astype(int)

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
                            whatever is desired by the swc. E.g. nm to microns has scaling=1000.
        '''

        skeleton_io.export_to_swc(self, filename, node_labels=node_labels,
                                  radius=radius, header=header, xyz_scaling=xyz_scaling)


