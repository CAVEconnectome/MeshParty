import numpy as np
from functools import partial
from scipy import interpolate
from scipy.spatial import KDTree


def make_windows(input_d):
    delta_d = np.abs(np.diff(input_d))
    d_min = input_d - np.concatenate([delta_d / 2, [0]])
    d_max = input_d + np.concatenate([[0], delta_d / 2])
    return d_min, d_max


def _assign_window(x, d_min, d_max):
    indv = np.flatnonzero(np.logical_and(x >= d_min, x < d_max))
    if len(indv) == 1:
        return indv[0]
    else:
        return None


def assign_windows(des_d, init_d):
    "Assign desired distances to windows determined from the initial distances for each vertex"
    d_min, d_max = make_windows(init_d)
    _p_assign_window = partial(_assign_window, d_min=d_min, d_max=d_max)
    _assign_window_v = np.vectorize(_p_assign_window)
    inds = _assign_window_v(des_d)
    return inds


def resample_path(
    path, sk, path_counter, spacing, kind, tip_length_ratio, branch_d, avoid_root
):
    "Resample a specific path to get a desired spacing"
    return_direct = False
    if path[-1] != sk.root:
        # find that last edge whose start point was the first vertex in the path
        # do this so we don't get big gaps from soma->branch or branch->branch
        last_node = int(sk.parent_nodes(path[-1]))
        if last_node == sk.root:
            mod_path = path
            if len(path) == 1 and avoid_root:
                return_direct = True
        else:
            path_as_list = list(path)
            path_as_list.append(last_node)
            mod_path = sk.SkeletonIndex(path_as_list)
        add_last_edge = True
    else:
        mod_path = path
        add_last_edge = False
        if len(mod_path) == 1 or (len(mod_path) == 2 and avoid_root):
            return_direct == True

    # If path only contains the root, or just one point and then the root and avoid_root is active,
    # just return without doing anything except updating new_edges.
    if return_direct:
        new_verts = sk.vertices[mod_path]
        if len(mod_path) == 1:
            new_edges = np.zeros((0, 2), dtype=int)
        else:
            new_edges = (
                np.vstack(
                    [
                        np.arange(len(new_verts) - 1, 0, -1, dtype=int),
                        np.arange(len(new_verts) - 2, -1, -1, dtype=int),
                    ]
                ).T
                + path_counter
            )
        if add_last_edge:
            new_edges = np.vstack([new_edges, [path_counter, branch_d[last_node]]])
        output_map_path = np.array(mod_path)
        return new_verts, new_edges, output_map_path, branch_d

    # use the distance from root to parameterize the path
    input_d = sk.distance_to_root[mod_path]

    # setup an interpolation function based upon distance to root as input and xyz as output
    fi = interpolate.interp1d(input_d, sk.vertices[mod_path, :], kind=kind, axis=0)

    # the desired distances from root are evenly spaced according to spacing
    des_d = np.arange(np.min(input_d), np.max(input_d), spacing)

    if avoid_root and mod_path[-1] == sk.root:
        # Use the tip length ratio or 1/2, whichever is larger to keep new nodes out of soma domain.
        # This keeps desired points out of domain of the root node.
        d_last = np.abs(input_d[-1] - input_d[-2])
        des_d = des_d[
            np.logical_or(
                des_d > (d_last * np.max((tip_length_ratio, 0.5))), des_d == 0
            )
        ]

    # use the function to interpolate the new values
    # we need to add that last node to the verts ONLY IF the last edge length
    # meets the cutoff defined in tip_length_ratio
    new_verts = fi(des_d)

    inds = assign_windows(des_d, input_d)
    output_map_path = np.array(mod_path[inds])

    tip_ind = mod_path[0]
    true_tip = sk.vertices[tip_ind, :]
    if np.linalg.norm(new_verts[-1] - true_tip) / spacing > tip_length_ratio:
        new_verts = np.vstack([new_verts, true_tip])
        output_map_path = np.concatenate((output_map_path, [tip_ind]))

    # find the index of the old branch points in the new path
    is_branch = np.isin(np.array(path), np.concatenate([sk.branch_points, np.array([sk.root])]))
    path_branch = np.array(path[is_branch])
    path_branch_verts = sk.vertices[path_branch, :]
    tree = KDTree(new_verts)
    map_ds, new_branch_on_path = tree.query(path_branch_verts)
    new_branch_on_path += path_counter
    # create a temporary dictionary with this path's branch  points
    new_branch_d = {pb: nw for pb, nw in zip(path_branch, new_branch_on_path)}
    # update the overall mapping dictionary
    branch_d.update(new_branch_d)

    # new edges just march down path from last vertex to first
    new_edges = (
        np.vstack(
            [
                np.arange(len(new_verts) - 1, 0, -1),
                np.arange(len(new_verts) - 2, -1, -1),
            ]
        ).T
        + path_counter
    )
    if add_last_edge:
        # if we do have one, then we want to add an edge that is the from the first vertex
        # in the path, to the new vertex (mapped through branch_d) of the edge we found
        # in the original edge list
        new_edges = np.vstack([new_edges, [path_counter, branch_d[last_node]]])
    return new_verts, new_edges, output_map_path, branch_d