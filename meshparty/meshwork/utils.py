import blosc
import numpy as np
import numba
from scipy import sparse
from ..trimesh_io import Mesh
from ..skeleton import Skeleton

DEFAULT_VOXEL_RESOLUTION = [4, 4, 40]


def MeshworkIndexFactory(mw):
    class JointMeshIndex(np.ndarray):
        def __new__(cls, mesh_indices):
            if (
                np.array(mesh_indices).dtype is np.dtype("bool")
                and len(mesh_indices) == mw.mesh.n_vertices
            ):
                mesh_indices = np.flatnonzero(mesh_indices)

            mesh_inds = np.asarray(mesh_indices, dtype=int)
            obj = mesh_inds.view(cls)
            obj._mesh_indices_base = mw.mesh.map_indices_to_unmasked(mesh_inds)
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self._mesh_indices_base = getattr(obj, "_mesh_indices_base", np.array([]))

        def __getitem__(self, k):
            ret = super(JointMeshIndex, self).__getitem__(k)
            if not isinstance(ret, np.integer):
                ret._mesh_indices_base = self._mesh_indices_base[k]
            return ret

        @property
        def to_mesh_index(self):
            return self

        @property
        def to_mesh_index_base(self):
            return self._mesh_indices_base

        @property
        def to_mesh_mask_base(self):
            mask = np.full(len(mw.mesh.node_mask), False)
            mask[self.to_mesh_index_base] = True
            return mask

        @property
        def to_mesh_mask(self):
            return mw.mesh.filter_unmasked_boolean(self.to_mesh_mask_base)

        @property
        def to_all_equivalent_mesh(self):
            return self.to_skel_index.to_mesh_index

        @property
        def to_all_equivalent_mask(self):
            return self.to_skel_index.to_mesh_mask

        @property
        def to_skel_index(self):
            if mw.skeleton is None:
                return None
            return JointSkeletonIndex(mw._mind_to_skind(self))

        @property
        def to_skel_index_padded(self):
            if mw.skeleton is None:
                return None
            return mw._mind_to_skind_padded(self)

        @property
        def to_skel_mask(self):
            if mw.skeleton is None:
                return None
            return mw._mesh_mask_to_skel_mask(self.to_mesh_mask)

    if mw.skeleton is None:
        return JointMeshIndex, np.array

    class JointSkeletonIndex(np.ndarray):
        def __new__(cls, skel_indices):
            if (
                np.array(skel_indices).dtype is np.dtype("bool")
                and len(skel_indices) == mw.skeleton.n_vertices
            ):
                skel_indices = np.flatnonzero(skel_indices)

            skel_inds = np.asarray(skel_indices, dtype=int)
            obj = skel_inds.view(cls)
            obj._skel_indices_base = mw.skeleton.map_indices_to_unmasked(
                skel_inds[skel_inds >= 0]
            )
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self._skel_indices_base = getattr(obj, "_skel_indices_base", np.array([]))

        def __getitem__(self, k):
            ret = super(JointSkeletonIndex, self).__getitem__(k)
            if not isinstance(ret, np.integer):
                ret._skel_indices_base = self._skel_indices_base[k]
            return ret

        @property
        def to_mesh_index(self):
            return JointMeshIndex(mw._skind_to_mind_index(self[self >= 0]))

        @property
        def to_mesh_index_base(self):
            return mw.mesh.map_indices_to_unmasked(self.to_mesh_index)

        @property
        def to_mesh_mask_base(self):
            return mw._skind_to_mind_mask_base(self[self >= 0])

        @property
        def to_mesh_mask(self):
            return mw._skind_to_mind_mask(self[self >= 0])

        @property
        def to_mesh_region(self):
            return [
                JointMeshIndex(x) if self[ii] >= 0 else JointMeshIndex([])
                for ii, x in enumerate(mw._skind_regions(self))
            ]

        @property
        def to_mesh_region_point(self):
            out = mw._skind_region_first(self)
            out[self < 0] = -1
            return JointMeshIndex(out)

        @property
        def to_skel_index(self):
            return self[self >= 0]

        @property
        def to_skel_index_padded(self):
            return self

        @property
        def to_skel_index_base(self):
            return self._skel_indices_base

        @property
        def to_skel_mask_base(self):
            mask = np.full(len(mw.skeleton.node_mask), False)
            mask[self._skel_indices_base] = True
            return mask

        @property
        def to_skel_mask(self):
            return mw.skeleton.filter_unmasked_boolean(self.to_skel_mask_base)

    return JointMeshIndex, JointSkeletonIndex


@numba.njit(parallel=True)
def _in1d_items(elements, mask, test_vals):
    maskinds = np.flatnonzero(mask)
    out = np.zeros((mask.sum(), 2), dtype=np.int64)
    for ii in numba.prange(len(maskinds)):
        val_ind = np.flatnonzero(elements[maskinds[ii]] == test_vals)[0]
        out[ii, 0] = np.int64(val_ind)
        out[ii, 1] = np.int64(maskinds[ii])
    return out


def in1d_items(elements, test_vals):
    """For each item in test_vals, finds all indices in elements that match it
    """
    out = _in1d_items(elements, np.isin(elements, test_vals), test_vals)
    return [out[out[:, 0] == ii, 1] for ii in range(len(test_vals))]


def in1d_first_item(elements, test_vals):
    """Given a long list of indices, finds one index for each of the indices
    in test_vals (or a -1)"""
    el_mask = np.isin(elements, test_vals)
    vals, ind_mask = np.unique(elements[el_mask], return_index=True)
    out = np.full(len(test_vals), -1)
    _, slots = np.unique(test_vals[np.isin(test_vals, vals)], return_inverse=True)
    out[slots] = np.flatnonzero(el_mask)[ind_mask]
    return out


# @numba.njit
# def _tree_density(
#     inds,
#     branch_points,
#     bp_downstream,
#     distance_to_root,
#     cover_paths,
#     width,
#     len_per,
#     weight,
# ):
#     norm = np.zeros(len(distance_to_root))
#     count = np.zeros(len(distance_to_root))
#     has_count = np.zeros(len(distance_to_root))
#     has_count[inds] = weight
#     has_bp = np.full(len(distance_to_root), -1)
#     has_bp[branch_points] = branch_points 

#     for path in cover_paths:
#         path_d = distance_to_root[path]
         
#         for jj in path:
#             dfrom = path_d - distance_to_root[jj]
#             window = np.abs(dfrom) <= width

#             norm[window] += len_per[jj]
#             count[jj] = has_count[path[window]].sum()

#             bps_in_window = has_bp[path[window]]
#             bps_in_window = bps_in_window[bps_in_window>=0]
#             for kk in bps_in_window:
#                 init_dist = distance_to_root[jj]-distance_to_root[kk]
#                 remaining_dist = width - init_dist
#                 bp_dist = distance_to_root[kk]
#                 inds_downstream = bp_downstream[kk] # Check if this is inclusive
#                 upstream_dist = distance_to_root[inds_downstream] - bp_dist
#                 inds_to_add = inds_downstream[upstream_dist <= remaining_dist]
                
#                 norm[inds_to_add] += len_per[jj]
#                 count[jj] += has_count[inds_to_add].sum()
#     return count, norm

def _window_matrix(branch_points, bp_downstream, distance_to_root, end_paths_to_root, width):
    """ Generate matrix such that Aij is 1 if j is within width of i.
    """
    has_bp = np.full(len(distance_to_root), -1)
    has_bp[branch_points] = branch_points 
    seen = np.full(len(distance_to_root), False)
    
    row_ind = []
    col_ind = []
    
    for path in end_paths_to_root:
        path_d = distance_to_root[path]
        for jj in path:
            dfrom = path_d - distance_to_root[jj]
            window = np.abs(dfrom) <= width
            bp_in_window = has_bp[path[window]]
            bp_in_window = bp_in_window[bp_in_window>=0]
            ind_list = [path[window]]
            for kk in bp_in_window:
                init_dist = np.abs(distance_to_root[jj]-distance_to_root[kk])
                remaining_dist = width - init_dist
                inds_downstream = bp_downstream[kk]
                upstream_dist = distance_to_root[inds_downstream] - distance_to_root[kk]
                inds_to_add = inds_downstream[upstream_dist <= remaining_dist]
                ind_list.append(inds_to_add)
            all_inds = np.unique(np.concatenate(tuple(ind_list)))
            for kk in all_inds:
                row_ind.append(jj)
                col_ind.append(kk)
            
            if seen[jj]:
                break
            else:
                seen[jj] = True
    return np.array(row_ind), np.array(col_ind)

def window_matrix(sk, width):
    """ Generate matrix such that Aij is 1 if j is within width of i.
    """
    bp_downstream = {bp: sk.downstream_nodes(bp) for bp in sk.branch_points}
    end_paths_to_root = [sk.path_to_root(ep) for ep in sk.end_points]  
    wm_inds = _window_matrix(sk.branch_points, bp_downstream, sk.distance_to_root, end_paths_to_root, width)
    return sparse.csr_matrix((np.ones(len(wm_inds[0])), wm_inds), shape=((sk.n_vertices, sk.n_vertices)))


def unique_column_name(base_name, suffix, df):
    if base_name is not None:
        col_name = f"{base_name}_{suffix}"
    else:
        col_name = suffix
    if col_name in df.columns:
        ii = 0
        while True:
            test_col_name = f"{col_name}_{ii}"
            ii += 1
            if test_col_name not in df.columns:
                col_name = test_col_name
                break
    return col_name


def compress_mesh_data(mesh, cname="lz4"):
    if mesh.voxel_scaling is not None:
        vxsc = mesh.voxel_scaling
    else:
        vxsc = None
    mesh.voxel_scaling = None
    zvs = blosc.compress(mesh.vertices.tostring(), typesize=8, cname=cname)
    zfs = blosc.compress(mesh.faces.tostring(), typesize=8, cname=cname)
    zes = blosc.compress(mesh.link_edges.tostring(), typesize=8, cname=cname)
    znm = blosc.compress(mesh.node_mask.tostring(), typesize=1, cname=cname)
    mesh.voxel_scaling = vxsc
    return zvs, zfs, zes, znm, vxsc


def decompress_mesh_data(zvs, zfs, zes, znm, vxsc):
    vs = np.frombuffer(blosc.decompress(zvs), dtype=np.float).reshape(-1, 3)
    fs = np.frombuffer(blosc.decompress(zfs), dtype=np.int).reshape(-1, 3)
    es = np.frombuffer(blosc.decompress(zes), dtype=np.int).reshape(-1, 2)
    nm = np.frombuffer(blosc.decompress(znm), dtype=np.bool)
    return vs, fs, es, nm, vxsc


class MaskedMeshMemory(object):
    def __init__(self, mesh, index_only=False):
        self.node_mask = mesh.node_mask.copy()
        self.map_indices_to_unmasked = mesh.map_indices_to_unmasked
        self.map_boolean_to_unmasked = mesh.map_boolean_to_unmasked
        self.filter_unmasked_boolean = mesh.filter_unmasked_boolean
        self.filter_unmasked_indices = mesh.filter_unmasked_indices
        self.filter_unmasked_indices_padded = mesh.filter_unmasked_indices_padded
        self._voxel_scaling = mesh.voxel_scaling
        if index_only is False:
            self._kdtree = mesh.kdtree

    @property
    def kdtree(self):
        return self._kdtree

    @property
    def voxel_scaling(self):
        return self._voxel_scaling
