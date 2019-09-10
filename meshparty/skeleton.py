import numpy as np
from meshparty import utils
from scipy import spatial, sparse
from pykdtree.kdtree import KDTree as pyKDTree
from copy import copy
import json
from meshparty import skeleton_io
from collections.abc import Iterable

class Skeleton:
    """Class to manage skeleton data
        
    Parameters
    ----------
    vertices : np.array
        a Nx3 list of xyz locations of skeleton nodes
    edges : np.array
        a Kx2 list of edges in the skeleton where each row is a connected pair of vertex indices.
    mesh_to_skel_map : np.array
        For a skeleton derived from a mesh with N_mesh vertices, a length N_mesh array where
        that gives the closest on-graph skeleton vertex for each mesh vertex (or -1 if None).
        (default is None)
    vertex_properties: dict
        a dictionary of keys with strings
        where each value is a numpy.array of len(N) of properties of skeleton vertices
    root : None
        what vertex index should be root (default None will find a vertex far from others) 
    """

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
        """ numpy.array : Nx3 set of xyz coordinates of skeletons"""
        return self._vertices

    @property
    def edges(self):
        """ numpy.array : Mx2 set of edges as indices into vertices """
        return self._edges

    @property
    def mesh_to_skel_map(self):
        """ numpy.array : N_mesh length array giving the associated skeleton
        vertex for each mesh vertex"""
        return self._mesh_to_skel_map
    
    @property
    def segments(self):
        """ list : A list of numpy.array indicies of segments, paths from each branch or
        end point (inclusive) to the next rootward branch/root point (exclusive), that
        cover the skeleton"""
        if self._segments is None:
            self._segments, self._segment_map = self._compute_segments()
        return self._segments

    @property
    def segment_map(self):
        """ np.array : N set of of indices between 0 and len(self.segments)-1, denoting
        which segment a given skeleton vertex is in.
        """
        if self._segment_map is None:
            self._segments, self._segment_map = self._compute_segments()
        return self._segment_map

    @property
    def csgraph(self):
        """ scipy.sparse.csr.csr_matrix : Directed sparse graph representation of the skeleton
        vertex connectivity. The i,jth element is the distance from a vertex i to its parent j,
        0 otherwise.
        """
        if self._csgraph is None:
            self._csgraph = self._create_csgraph()
        return self._csgraph.copy()

    @property
    def csgraph_binary(self):
        """ scipy.sparse.csr.csr_matrix : Directed binary sparse graph representation of
        skeleton vertex connectivity. The i,jth element is 1 if vertex i has parent j, 0 otherwise.
        """
        if self._csgraph_binary is None:
            self._csgraph_binary = self._create_csgraph(euclidean_weight=False)
        return self._csgraph_binary

    @property
    def csgraph_undirected(self):
        """ scipy.sparse.csr_matrix : Undirected sparse graph representation of skeleton
        vertex connectivity. The i,jth element is the distance from vertex i to any connected
        vertex j, 0 otherwise.
        """
        return self.csgraph + self.csgraph.T

    @property
    def csgraph_binary_undirected(self):
        """ scipy.sparse.csr_matrix : Undirected sparse graph representation of skeleton
        vertex connectivity. The i,jth element is 1 if vertex i is connected to vertex j, 0 otherwise.
        """
        return self.csgraph_binary + self.csgraph_binary.T

    @property
    def n_vertices(self):
        """ int : Number of vertices in the skeleton """
        return len(self.vertices)

    @property
    def root(self):
        """ int : Index of the skeleton root """
        if self._root is None:
            self._create_default_root()
        return copy(self._root)

    @property
    def pykdtree(self):
        """ pykdtree.pyKDTree object : k-D tree from pykdtree (a bit faster but fewer functions) """
        if self._pykdtree is None:
            self._pykdtree = pyKDTree(self.vertices)
        return self._pykdtree

    @property
    def kdtree(self):
        """ scipy.spatial.kdtree : k-D tree from scipy.spatial. """
        if self._kdtree is None:
            self._kdtree = spatial.cKDTree(self.vertices)
        return self._kdtree
    

    @property
    def branch_points(self):
        """ numpy.array : Indices of branch points on the skeleton (pottentially including root)"""  
        if self._branch_points is None:
            self._create_branch_and_end_points()
        return self._branch_points.copy()

    @property
    def n_branch_points(self):
        """ int : Number of branch points on the skeleton """
        if self._branch_points is None:
            self._create_branch_and_end_points()
        return len(self._branch_points)

    @property
    def end_points(self):
        """ numpy.array : Indices of end points on the skeleton (pottentially including root)"""
        if self._end_points is None:
            self._create_branch_and_end_points()
        return self._end_points.copy()

    @property
    def n_end_points(self):
        """ int : Number of end points on the skeleton """
        if self._end_points is None:
            self._create_branch_and_end_points()
        return len(self._end_points)

    @property
    def cover_paths(self):
        """ list : List of numpy.array objects with self.n_end_points elements, each a rootward
        path (ordered set of indices) starting from an endpoint and continuing until it reaches
        a point on a path earlier on the list. Paths are ordered by end point distance from root, 
        starting with the most distal. When traversed from begining to end, gives the longest rootward 
        path down each major branch first. When traversed from end to begining, guarentees every 
        branch point is visited at the end of all its downstream paths before being traversed along
        a path.
        """
        if self._cover_paths is None:
            self._cover_paths = self._compute_cover_paths()
        return self._cover_paths

    @property
    def distance_to_root(self):
        """ np.array : N length array with the distance to the root node along the skeleton. 
        """
        ds = sparse.csgraph.dijkstra(self.csgraph, directed=False,
                                     indices=self.root)
        return ds

    def path_to_root(self, v_ind):
        '''
        Gives the path to root from a specified vertex.

        Parameters
        ----------
        v_ind : int
            Vertex index
        
        Returns
        -------
        numpy.array : Ordered set of indices from v_ind to root, inclusive of both.
        '''
        path = [v_ind]
        ind = v_ind
        ind = self.parent_node(ind)
        while ind is not None:
            path.append(ind)
            ind = self.parent_node(ind)
        return np.array(path)

    def path_length(self, paths):
        """Returns the length of a path (described as an ordered collection of connected indices)
        Parameters
        ----------
        path : Path or collection of paths, a path being an ordered list of linked vertex indices.

        Returns
        -------
        Float or list of floats : The length of each path.
        """
        if len(paths) == 0:
            return 0
        
        if isinstance(paths[0], Iterable):
            Ls = []
            for path in paths:
                Ls.append(self._single_path_length(path))
        else:
            Ls = self._single_path_length(paths)
        return Ls 

    def reroot(self, new_root):
        """Change the skeleton root index.
        
        Parameters
        ----------
        new_root : Int
            Skeleton vertex index to be the new root.
        """
        if new_root > self.n_vertices:
            raise ValueError('New root must correspond to a skeleton vertex index')
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
        """Return a csgraph for the skeleton with specified vertices cut off from their parent vertex.
        
        Parameters
        ----------
        vinds :  
            Collection of indices to cut off from their parent.
        directed : bool, optional
            Return the graph as directed, by default True
        euclidean_weight : bool, optional
            Return the graph with euclidean weights, by default True. If false, the unweighted.
        
        Returns
        -------
        scipy.sparse.csr.csr_matrix  
            Graph with vertices in vinds cutt off from parents.
        """
        e_keep = ~np.isin(self.edges[:,0], vinds)
        es_new = self.edges[e_keep]
        return utils.create_csgraph(self.vertices, es_new,
                                    euclidean_weight=euclidean_weight,
                                    directed=directed)


    def downstream_nodes(self, vinds):
        """Get a list of all nodes downstream of a collection of indices
        
        Parameters
        ----------
        vinds : Collection of ints
            Collection of vertex indices
        
        Returns
        -------
        List of arrays
            List whose ith element is an array of all vertices downstream of the ith element of vinds.
        """
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
        """Get a list of all immediate children of list of indices
        
        Parameters
        ----------
        vinds : Collection of ints
            Collection of vertex indices
        
        Returns
        -------
        List of arrays
            A list whose ith element is the children of the ith element of vinds. 
        """
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
        """Get a list of parent nodes for specified vertices 
        
        Parameters
        ----------
        vinds : Collection of ints
            Collection of vertex indices
        
        Returns
        -------
        numpy.array
            The parent node of each vertex index in vinds.
        """
        if isinstance(vinds, list):
            vinds = np.array(vinds)
        return self._parent_node_array[vinds]

    def _reset_derived_objects(self):
        """Reset all properties that could change when skeletons are rerooted.
        """
        self._csgraph = None
        self._csgraph_binary = None
        self._cover_paths = None
        self._segments = None
        self._segment_map = None

    def _create_csgraph(self,
                        directed=True,
                        euclidean_weight=True):
        """Create the csgraph for the skeleton.
        """
        return utils.create_csgraph(self.vertices, self.edges,
                                    euclidean_weight=euclidean_weight,
                                    directed=directed)

    def _create_branch_and_end_points(self):
        """Pre-compute branch and end points from the graph
        """
        n_children = np.sum(self.csgraph_binary > 0, axis=0).squeeze()
        self._branch_points = np.flatnonzero(n_children > 1)
        self._end_points = np.flatnonzero(n_children == 0)

    def _single_path_length(self, path):
        """Compute the length of a single path
        """
        xs = self.vertices[path[:-1]]
        ys = self.vertices[path[1:]]
        return np.sum(np.linalg.norm(ys-xs, axis=1))

    def _compute_cover_paths(self):
        '''Compute the list of cover paths along the skeleton
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
        """Precompute segments between branches and end points"""
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

        Parameters
        ----------
        filename : str
            Name of the file to save the swc to
        node_labels : np.array
            None (default) or an interable of ints co-indexed with vertices.
            Corresponds to the swc node categories. Defaults to setting all
            odes to label 3, dendrite.
        radius : iterable
            None (default) or an iterable of floats. This should be co-indexed with vertices.
            Radius values are assumed to be in the same units as the node vertices.
        header : dict
            default None. Each key value pair in the dict becomes
            a parameter line in the swc header.
        xyz_scaling : Number
            default 1000. Down-scales spatial units from the skeleton's units to
            whatever is desired by the swc. E.g. nm to microns has scaling=1000.
            
        '''

        skeleton_io.export_to_swc(self, filename, node_labels=node_labels,
                                  radius=radius, header=header, xyz_scaling=xyz_scaling)


