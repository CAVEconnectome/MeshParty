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

def edge_averaged_vertex_property(edge_property, vertices, edges):
    vertex_property = np.full((len(vertices),2), np.nan)
    vertex_property[edges[:,0],0] = np.array(edge_property)
    vertex_property[edges[:,1],1] = np.array(edge_property)
    return np.nanmean(vertex_property, axis=1)
    
def reduce_vertices(vertices, edges, v_filter=None, e_filter=None, return_filter_inds=False):
    if v_filter is None:
        v_filter = np.unique(edges).astype(int)
    if e_filter is None:
        e_filter_bool = np.isin(edges[:,0], v_filter) & np.isin(edges[:,1], v_filter)
        e_filter = np.where(e_filter_bool)[0]
    
    vertices_n = vertices[v_filter]
    vmap = dict(zip(v_filter, np.arange(len(v_filter))))
    
    edges_f = edges[e_filter].copy()
    edges_n = np.stack((np.fromiter((vmap[x] for x in edges_f[:,0]), dtype=int),
                        np.fromiter((vmap[x] for x in edges_f[:,1]), dtype=int))).T

    if return_filter_inds:
        return vertices_n, edges_n, (v_filter, e_filter)
    else:
        return vertice_n, edges_n

def create_csgraph(vertices,
                   edges,
                   euclidean_weight=True):

    if euclidean_weight:
        xs = vertices[edges[:,0]]
        ys = vertices[edges[:,1]]
        weights = np.linalg.norm(xs-ys, axis=1)
        use_dtype = np.float32
    else:   
        weights = np.ones((len(edges),)).astype(bool)
        use_dtype = bool 

    edges = np.concatenate([edges.T, edges.T[[1, 0]]], axis=1)
    weights = np.concatenate([weights, weights]).astype(dtype=use_dtype)

    csgraph = sparse.csr_matrix((weights, edges),
                                shape=[len(vertices), ] * 2,
                                dtype=use_dtype)

    return csgraph


class SkeletonForest():
    def __init__(self, vertices, edges, vertex_properties={}, edge_properties={}, root=None):
        self._vertex_components = np.full(len(vertices), None)
        self._edge_components = np.full(len(edges), None)
        self._skeletons = []
        self._kdtree = None

        vertices = np.array(vertices)
        edges = np.array(edges)

        nc, v_lbls = sparse.csgraph.connected_components(create_csgraph(vertices, edges, euclidean_weight=False))
        lbls, count = np.unique(v_lbls, return_counts=True)
        lbl_order = np.argsort(count)[::-1]
        for lbl in lbls[lbl_order]:
            v_filter = np.where(v_lbls==lbl)[0]
            vertices_f, edges_f, filters = reduce_vertices(vertices,
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
        return create_csgraph(self.vertices_all, self.edges_all)

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
        r = find_far_points_graph(self._create_csgraph(directed=False))
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


def connected_component_slice(G, ind=None):
    _, labels = sparse.csgraph.connected_components(G)
    if ind is None:
        label_vals, cnt = np.unique(labels, return_counts=True)
        ind = np.argmax(cnt)
        label = label_vals[ind]
    else:
        label = labels[ind]
    return np.where(labels == label)[0]


def get_path(root, target, pred):
    path = [target]
    p = target
    while p != root:
        p = pred[p]
        path.append(p)
    path.reverse()
    return path


def find_far_points(trimesh, multicomponent=False):
    return find_far_points_graph(trimesh.csgraph)


def find_far_points_graph(mesh_graph, multicomponent=False):
    d = 0
    dn = 1
   
    if multicomponent: 
        a = connected_component_slice(mesh_graph)[0]
    else:
        a = 0
    b = 1
    
    k = 0
    pred = None
    ds = None
    while 1:
        k += 1
        dsn, predn = sparse.csgraph.dijkstra(
            mesh_graph, False, a, return_predecessors=True)
        if multicomponent:
            dsn[np.isinf(dsn)] = 0
        bn = np.argmax(dsn)
        dn = dsn[bn]
        if dn > d:
            b = a
            a = bn
            d = dn
            pred = predn
            ds = dsn
        else:
            break

    return b, a, pred, d, ds



def setup_root(mesh, soma_pos=None, soma_thresh=7500):
    valid = np.ones(len(mesh.vertices), np.bool)

    # soma mode
    if soma_pos is not None:
        # pick the first soma as root
        soma_d, soma_i = mesh.kdtree.query(soma_pos,
                                           k=len(mesh.vertices),
                                           distance_upper_bound=soma_thresh)
        assert(np.min(soma_d)<soma_thresh)
        root = soma_i[np.argmin(soma_d)]
        valid[soma_i[~np.isinf(soma_d)]] = False
        root_ds, pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                False,
                                                root,
                                                return_predecessors=True)
    else:
        # there is no soma so use far point heuristic
        root, target, pred, dm, root_ds = find_far_points(mesh)
        valid[root] = 0

    return root, root_ds, pred, valid


def recenter_verts(verts, edges, centers):
    edge_df = pd.DataFrame()
    edge_df['start_edge']=np.array(edges[:,0], np.int64)
    edge_df['end_edge']=np.array(edges[:,1], np.int64)
    edge_df['center_x']=np.array(centers)[:,0]
    edge_df['center_y']=np.array(centers)[:,1]
    edge_df['center_z']=np.array(centers)[:,2]
    start_mean = edge_df.groupby('start_edge').mean()[['center_x','center_y','center_z']]
    new_verts = np.copy(verts)
    new_verts[start_mean.index.values,:]=start_mean.values
    return new_verts


def skeletonize(mesh_meta, seg_id, soma_pt=None, soma_thresh=7500,
                invalidation_d=10000, smooth_neighborhood=5,
                max_tip_d=2000, large_skel_path_threshold=5000,
                cc_vertex_thresh=100, do_cross_section=False):

    mesh = mesh_meta.mesh(seg_id=seg_id,
                          merge_large_components=False,
                          remove_duplicate_vertices=False)
    mesh.stitch_overlapped_components()

    all_paths, roots, tot_path_lengths = skeletonize_components(mesh,
                                                                soma_pt=soma_pt,
                                                                soma_thresh=soma_thresh,
                                                                invalidation_d=invalidation_d,
                                                                cc_vertex_thresh=cc_vertex_thresh)
    tot_edges = merge_tips(mesh, all_paths, roots, tot_path_lengths,
                           large_skel_path_threshold=large_skel_path_threshold, max_tip_d=max_tip_d)

    skel_verts, skel_edges = trimesh_vtk.remove_unused_verts(mesh.vertices, tot_edges)
    smooth_verts = smooth_graph(skel_verts, skel_edges, neighborhood=smooth_neighborhood)
    if do_cross_section:
        cross_sections, centers = trimesh_vtk.calculate_cross_sections(mesh, smooth_verts, skel_edges)
        center_verts = recenter_verts(smooth_verts, skel_edges, centers)
        smooth_center_verts = smooth_graph(center_verts, skel_edges, neighborhood=smooth_neighborhood)
        return skel_verts, skel_edges, cross_sections, smooth_verts, smooth_center_verts
    else:
        return skel_verts, skel_edges, smooth_verts


def skeletonize_axon(mesh_meta, axon_id, invalidation_d=5000, smooth_neighborhood=5,
                     max_tip_d=2000, large_skel_path_threshold=5000, cc_vertex_thresh=100):
    return skeletonize(mesh_meta, axon_id,
                       invalidation_d=invalidation_d,
                       smooth_neighborhood=smooth_neighborhood,
                       max_tip_d=max_tip_d,
                       large_skel_path_threshold=large_skel_path_threshold,
                       cc_vertex_thresh=cc_vertex_thresh)


def merge_tips(mesh, all_paths, roots, tot_path_lengths,
               large_skel_path_threshold=5000, max_tip_d=2000):

    # collect all the tips of the skeletons (including roots)
    skel_tips = []
    all_tip_indices = []
    for paths, root in zip(all_paths, roots):
        tips = []
        tip_indices = []
        for path in paths:
            tip_ind = path[0]
            tip = mesh.vertices[tip_ind, :]
            tips.append(tip)
            tip_indices.append(tip_ind)
        root_tip = mesh.vertices[root, :]
        tips.append(root_tip)
        tip_indices.append(root)
        skel_tips.append(np.vstack(tips))
        all_tip_indices.append(np.array(tip_indices))
    # this is our overall tip matrix merged together
    all_tips = np.vstack(skel_tips)
    # and the vertex index of those tips in the original mesh
    all_tip_indices = np.concatenate(all_tip_indices)
    
    # variable to keep track of what component each tip was from
    tip_component = np.zeros(all_tips.shape[0])
    # counter to keep track of an overall tip index as we go through 
    # the components with different numbers of tips
    ind_counter = 0

    # setup the prize collection steiner forest problem variables
    # prizes will be related to path length of the tip components
    tip_prizes = [] 
    # where to collect all the tip<>tip edges
    all_edges = []
    # where to collect all the tip<>tip edge weights
    all_edge_weights = []

    # loop over all the components and their tips
    for k, tips, path_lengths in zip(range(len(tot_path_lengths)), skel_tips, tot_path_lengths):
        # how many tips in this component
        ntips = tips.shape[0]
        # calculate the total path length in this component
        path_len = np.sum(np.array(path_lengths))
        # the prize is 0 if this is small, and the path length if big
        prize = path_len if path_len > large_skel_path_threshold else 0
        # the cost of traveling within a skeleton is 0 if big, and the path_len if small
        cost = path_len if path_len <= large_skel_path_threshold else 0
        # add a block of prizes to the tip prizes for this component
        tip_prizes.append(prize*np.ones(ntips))
        # make an array of overall tip index for this component
        comp_tips = np.arange(ind_counter, ind_counter+ntips, dtype=np.int64)
        # add edges between this components root and each of the tips
        root_tips = (ind_counter+ntips-1)*np.ones(ntips, dtype=np.int64)
        in_tip_edges = np.hstack([root_tips[:, np.newaxis],
                                  comp_tips[:, np.newaxis]])
        all_edges.append(in_tip_edges)

        # add a block for the cost of these edges
        all_edge_weights.append(cost*np.ones(ntips))
        # note what component each of these tips is from
        tip_component[comp_tips] = k
        # increment our overall index counter
        ind_counter += ntips
    # gather all the prizes into a single block
    tip_prizes = np.concatenate(tip_prizes)

    # make a kdtree with all the tips
    tip_tree = KDTree(all_tips)
    
    # find the tips near one another
    close_tips = tip_tree.query_pairs(max_tip_d, output_type='ndarray')
    # filter out close tips from the same component
    diff_comp = ~(tip_component[close_tips[:, 0]] == tip_component[close_tips[:, 1]])
    filt_close_tips = close_tips[diff_comp]

    # add these as edges
    all_edges.append(filt_close_tips)
    # with weights equal to their euclidean distance
    dv = np.linalg.norm(all_tips[filt_close_tips[:,0],:] - all_tips[filt_close_tips[:, 1]], axis=1)
    all_edge_weights.append(dv)

    # consolidate the edges and weights into a single array
    inter_tip_weights = np.concatenate(all_edge_weights)
    inter_tip_edges = np.concatenate(all_edges)

    # run the prize collecting steiner forest optimization
    mst_verts, mst_edges = pcst_fast.pcst_fast(
        inter_tip_edges, tip_prizes, inter_tip_weights, -1, 1, 'gw', 1)
#     # find the set of mst edges that are between connected components
    new_mst_edges = mst_edges[tip_component[inter_tip_edges[mst_edges, 0]] != tip_component[inter_tip_edges[mst_edges ,1]]]
    good_inter_tip_edges = inter_tip_edges[new_mst_edges, :]
    # get these in the original index
    new_edges_orig_ind = all_tip_indices[good_inter_tip_edges]
#     # collect all the edges for all the paths into a single list
#     # with the original indices of the mesh
    orig_edges = []
    for paths, root in zip(all_paths, roots):
        edges = paths_to_edges(paths)
        orig_edges.append(edges)
    orig_edges = np.vstack(orig_edges)
    # and add our new mst edges
    tot_edges = np.vstack([orig_edges, new_edges_orig_ind])

    return tot_edges


# def fix_skeleton(verts, edges):
#     # fix the skeleton so that it    
#     # filter out the vertices to be just those included in this skeleton
#     skel_verts, skel_edges = trimesh_vtk.remove_unused_verts(axon_trimesh.vertices, tot_edges)
#     Nind = skel_verts.shape[0]
#     g=sparse.csc_matrix((np.ones(len(skel_edges)), (skel_edges[:,0], skel_edges[:,1])), shape =
#                         (Nind,Nind))

#     # figure out how many skeleton components we have now
#     n_skel_comp, skel_labels = sparse.csgraph.connected_components(g,directed=False, return_labels=True)
#     skel_comp_labels, skel_comp_counts = np.unique(skel_labels, return_counts = True)
#     larg_skel_cc_ind = np.where(skel_comp_counts>100)[0]
#     large_skel_components=len(larg_skel_cc_ind)


def skeletonize_components(mesh, soma_pt=None, soma_thresh=10000, invalidation_d=10000, cc_vertex_thresh=100):
    # find all the connected components in the mesh
    n_components, labels = sparse.csgraph.connected_components(mesh.csgraph,
                                                               directed=False,
                                                               return_labels=True)
    comp_labels, comp_counts = np.unique(labels, return_counts=True)

    # variables to collect the paths, roots and path lengths
    all_paths = []
    roots = []
    tot_path_lengths = []
    
    if soma_pt is not None:
        soma_d = mesh.vertices - soma_pt[np.newaxis, :]
        soma_d = np.linalg.norm(soma_d, axis=1)
        is_soma_pt = soma_d<7500
    else:
        is_soma_pt = None
        soma_d = None
    # loop over the components
    for k in trange(n_components):
        if comp_counts[k] > cc_vertex_thresh:

            # find the root using a soma position if you have it
            # it will fall back to a heuristic if the soma
            # is too far away for this component
            root, root_ds, pred, valid = setup_root_new(mesh,
                                                        is_soma_pt,
                                                        soma_d,
                                                        labels == k)

            # run teasar on this component
            paths, path_lengths = mesh_teasar(mesh,
                                              root=root,
                                              root_ds=root_ds,
                                              root_pred=pred,
                                              valid=valid,
                                              invalidation_d=invalidation_d)
  
            if len(path_lengths) > 0:
                # collect the results in lists
                tot_path_lengths.append(path_lengths)
                all_paths.append(paths)
                roots.append(root)
            

    return all_paths, roots, tot_path_lengths


def setup_root_new(mesh, is_soma_pt=None, soma_d=None, is_valid=None):
    if is_valid is not None:
        valid = np.copy(is_valid)
    else:
        valid = np.ones(len(mesh.vertices), np.bool)
    root = None
    # soma mode
    if is_soma_pt is not None:
        # pick the first soma as root
        
        is_valid_root = is_soma_pt & valid
        valid_root_inds = np.where(is_valid_root)[0]
        if len(valid_root_inds) > 0:
            min_valid_root = np.nanargmin(soma_d[valid_root_inds])
            root = valid_root_inds[min_valid_root]
            root_ds, pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                    False,
                                                    root,
                                                    return_predecessors=True)
        else:
            start_ind = np.where(valid)[0][0]
            root, target, pred, dm, root_ds = find_far_points(mesh,
                                                              start_ind=start_ind)
        valid[is_soma_pt] = False

    if root is None:
        # there is no soma close, so use far point heuristic
        start_ind = np.where(valid)[0][0]
        root, target, pred, dm, root_ds = find_far_points(mesh, start_ind=start_ind)
    valid[root] = False
    assert(np.all(~np.isinf(root_ds[valid])))
    return root, root_ds, pred, valid


def setup_root(mesh, soma_pt=None, soma_thresh=7500, valid_inds=None):
    if valid_inds is not None:
        valid = np.zeros(len(mesh.vertices), np.bool)
        valid[valid_inds] = True
    else:
        valid = np.ones(len(mesh.vertices), np.bool)
    root = None
    # soma mode
    if soma_pt is not None:
        # pick the first soma as root
        soma_d, soma_i = mesh.kdtree.query(soma_pt,
                                           k=len(mesh.vertices),
                                           distance_upper_bound=soma_thresh)
        if valid_inds is not None:
            soma_i, valid_soma_ind, soma_valid_ind = np.intersect1d(soma_i,
                                                                    valid_inds,
                                                                    return_indices=True)
            soma_d = soma_d[valid_soma_ind]
        if (len(soma_d) > 0):
            min_d = np.min(soma_d)
        else:
            min_d = np.inf 
        if (min_d < soma_thresh):
            root = soma_i[np.argmin(soma_d)]
            root_ds, pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                    False,
                                                    root,
                                                    return_predecessors=True)
        else:
            if valid_inds is not None:
                root, target, pred, dm, root_ds = find_far_points(mesh, start_ind=valid_inds[0])
            else:
                root, target, pred, dm, root_ds = find_far_points(mesh)
    if root is None:
        # there is no soma close, so use far point heuristic
        root, target, pred, dm, root_ds = find_far_points(mesh)
    valid[root] = False

    return root, root_ds, pred, valid


def mesh_teasar(mesh, root=None, valid=None, root_ds=None, root_pred=None, soma_pt=None,
                soma_thresh=7500, invalidation_d=10000, return_timing=False):
    # if no root passed, then calculation one
    if root is None:
        root, root_ds, root_pred, valid = setup_root(mesh,
                                                     soma_pt=soma_pt,
                                                     soma_thresh=soma_thresh)
    # if root_ds have not be precalculated do so
    if root_ds is None:
        root_ds, root_pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                     False,
                                                     root,
                                                     return_predecessors=True)
    # if certain vertices haven't been pre-invalidated start with just
    # the root vertex invalidated
    if valid is None:
        valid = np.ones(len(mesh.vertices), np.bool)
        valid[root] = False
    else:
        if (len(valid) != len(mesh.vertices)):
            raise Exception("valid must be length of vertices")

    total_to_visit = np.sum(valid)
    if np.sum(np.isinf(root_ds) & valid) != 0:
        print(np.where(np.isinf(root_ds) & valid))
        raise Exception("all valid vertices should be reachable from root")

    # vector to store each branch result
    paths = []

    # vector to store each path's total length
    path_lengths = []

    # keep track of the nodes that have been visited
    visited_nodes = [root]

    # counter to track how many branches have been counted
    k = 0

    # arrays to track timing
    start = time.time()
    time_arrays = [[], [], [], [], []]

    with tqdm(total=total_to_visit) as pbar:
        # keep looping till all vertices have been invalidated
        while(np.sum(valid) > 0):
            k += 1
            t = time.time()
            # find the next target, farthest vertex from root
            # that has not been invalidated
            target = np.nanargmax(root_ds*valid)
            if (np.isinf(root_ds[target])):
                raise Exception('target cannot be reached')
            time_arrays[0].append(time.time()-t)

            t = time.time()
            # figure out the longest this branch could be
            # by following the route from target to the root
            # and finding the first already visited node (max_branch)
            # The dist(root->target) - dist(root->max_branch)
            # is the maximum distance the shortest route to a branch
            # point from the target could possibly be,
            # use this bound to reduce the djisktra search radius for this target
            max_branch = target
            while max_branch not in visited_nodes:
                max_branch = root_pred[max_branch]
            max_path_length = root_ds[target]-root_ds[max_branch]

            # calculate the shortest path to that vertex
            # from all other vertices
            # up till the distance to the root
            ds, pred_t = sparse.csgraph.dijkstra(
                mesh.csgraph,
                False,
                target,
                limit=max_path_length,
                return_predecessors=True)

            # pick out the vertex that has already been visited
            # which has the shortest path to target
            min_node = np.argmin(ds[visited_nodes])
            # reindex to get its absolute index
            branch = visited_nodes[min_node]
            # this is in the index of the point on the skeleton
            # we want this branch to connect to
            time_arrays[1].append(time.time()-t)

            t = time.time()
            # get the path from the target to branch point
            path = get_path(target, branch, pred_t)
            visited_nodes += path[0:-1]
            # record its length
            assert(~np.isinf(ds[branch]))
            path_lengths.append(ds[branch])
            # record the path
            paths.append(path)
            time_arrays[2].append(time.time()-t)

            t = time.time()
            # get the distance to all points along the new path
            # within the invalidation distance
            dm = sparse.csgraph.dijkstra(
                mesh.csgraph, False, path, limit=invalidation_d, min_only=True)
            time_arrays[3].append(time.time()-t)

            t = time.time()
            # all such non infinite distances are within the invalidation
            # zone and should be marked invalid
            marked = np.sum(valid & ~np.isinf(dm))
            valid[~np.isinf(dm)] = False
            # print out how many vertices are still valid
            pbar.update(marked)
            time_arrays[4].append(time.time()-t)
    # record the total time
    dt = time.time() - start

    if return_timing:
        return paths, path_lengths, time_arrays, dt
    else:
        return paths, path_lengths


def paths_to_edges(path_list):
    arrays = []
    for path in path_list:
        p = np.array(path)
        e = np.vstack((p[0:-1], p[1:])).T
        arrays.append(e)
    return np.vstack(arrays)


def smooth_graph(verts, edges, neighborhood=2, iterations=100, r=.1):
    """ smooths a spatial graph via iterative local averaging
        calculates the average position of neighboring vertices
        and relaxes the vertices toward that average

        :param verts: a NxK numpy array of vertex positions
        :param edges: a Mx2 numpy array of vertex indices that are edges
        :param neighborhood: an integer of how far in the graph to relax over
        as being local to any vertex (default = 2)
        :param iterations: number of relaxation iterations (default = 100)
        :param r: relaxation factor at each iteration
        new_vertex = (1-r)*old_vertex + r*(local_avg)

        :return: new_verts
        verts is a Nx3 list of new smoothed vertex positions

    """
    N = len(verts)
    E = len(edges)
    # setup a sparse matrix with the edges
    sm = sparse.csc_matrix(
        (np.ones(E), (edges[:, 0], edges[:, 1])), shape=(N, N))

    # an identity matrix
    eye = sparse.csc_matrix((np.ones(N, dtype=np.float32),
                                    (np.arange(0, N), np.arange(0, N))),
                                    shape=(N, N))
    # for undirected graphs we want it symettric
    sm = sm + sm.T

    # this will store our relaxation matrix
    C = sparse.csc_matrix(eye)
    # multiple the matrix and add to itself
    # to spread connectivity along the graph
    for i in range(neighborhood):
        C = C + sm @ C
    # zero out the diagonal elements
    C.setdiag(np.zeros(N))
    # don't overweight things that are connected in more than one way
    C = C.sign()
    # measure total effective neighbors per node
    neighbors = np.sum(C, axis=1)

    # normalize the weights of neighbors according to number of neighbors
    neighbors = 1/neighbors
    C = C.multiply(neighbors)
    # convert back to csc
    C = C.tocsc()

    # multiply weights by relaxation term
    C *= r

    # construct relaxation matrix, adding on identity with complementary weight
    A = C + (1-r)*eye

    # make a copy of original vertices to no destroy inpuyt
    new_verts = np.copy(verts)

    # iteratively relax the vertices
    for i in range(iterations):
        new_verts = A*new_verts
    return new_verts


def collapse_soma_skeleton(soma_pt, verts, edges, soma_d_thresh=12000):
    if soma_pt is not None:
        soma_pt_m = soma_pt[np.newaxis, :]
        dv = np.linalg.norm(verts - soma_pt_m, axis=1)
        soma_verts = np.where(dv < soma_d_thresh)[0]
        new_verts = np.vstack((verts, soma_pt_m))
        soma_i = verts.shape[0]
        edges[np.isin(edges, soma_verts)] = soma_i
        simple_verts, simple_edges = trimesh_vtk.remove_unused_verts(new_verts, edges)
        good_edges = ~(simple_edges[:, 0] == simple_edges[:, 1])
        return simple_verts, simple_edges[good_edges]
    else:
        simple_verts, simple_edges = trimesh_vtk.remove_unused_verts(verts, edges)
        return simple_verts, simple_edges
