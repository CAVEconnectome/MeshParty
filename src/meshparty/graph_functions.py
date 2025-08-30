import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from scipy import sparse
from .utils import single_path_length, build_csgraph


# Cache containers for expensive computations
@dataclass
class DAGCache:
    """
    Container for cached DAG properties to optimize repeated computations.
    Cache values are always in positional indices.

    Attributes
    ----------
    topological_order : Optional[List[int]]
        Cached topological ordering of vertices, computed once and reused.
    depths : Optional[Dict[int, int]]
        Cached depths of each vertex from root nodes.
    ancestors : Dict[int, Dict[int, int]]
        Binary lifting table where ancestors[v][i] = 2^i-th ancestor of vertex v.
        Used for efficient LCA queries in O(log n) time.
    parent_node_array : Optional[np.ndarray]
        Cached parent nodes for each vertex.
    branch_points : Optional[np.ndarray]
        Cached branch points (vertices with multiple parents).
    end_points : Optional[np.ndarray]
        Cached end points (leaf vertices with no children).

    """

    topological_order: Optional[List[int]] = None
    depths: Optional[Dict[int, int]] = None
    ancestors: Dict[int, Dict[int, int]] = None
    parent_node_array: Optional[np.ndarray] = None
    branch_points: Optional[np.ndarray] = None
    end_points: Optional[np.ndarray] = None
    segments: Optional[List[List[int]]] = None
    distance_to_root: Optional[np.ndarray] = None
    hops_to_root: Optional[np.ndarray] = None
    cover_paths: Optional[List[List[int]]] = None
    root: Optional[int] = None

    def __post_init__(self):
        """Initialize mutable default values to avoid shared state."""
        if self.ancestors is None:
            self.ancestors = {}

    def reset(self):
        """Reset all cached properties."""
        self.topological_order = None
        self.depths = None
        self.ancestors = {}
        self.parent_node_array = None
        self.branch_points = None
        self.end_points = None
        self.segments = None
        self.root = None
        self.distance_to_root = None
        self.hops_to_root = None
        self.cover_paths = None


def build_parent_node_array(vertices, edges) -> np.ndarray:
    """Assumes edges are in the form of [child_id, parent_id] pairs."""
    parent_node_array = np.full(len(vertices), -1, dtype=int)
    for e in edges:
        parent_node_array[e[0]] = e[1]
    return parent_node_array


def build_child_node_dictionary(vertices, csgraph) -> dict:
    # Get unique values and inverse mapping
    _, col_idx = csgraph.nonzero()
    unique_vals, inverse_indices = np.unique(col_idx, return_inverse=True)

    # Create dictionary
    col_dict = {}
    for i, unique_val in enumerate(unique_vals):
        if unique_val in vertices:
            col_dict[unique_val] = np.where(inverse_indices == i)[0]
    return col_dict


def build_adjacency_lists(
    edges: Union[np.ndarray, List[List[int]]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Build forward and reverse adjacency lists from edge array.

    Parameters
    ----------
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.

    Returns
    -------
    Tuple[Dict[int, List[int]], Dict[int, List[int]]]
        A tuple containing:
        - forward_adj: Dictionary mapping parent_id -> list of child_ids
        - reverse_adj: Dictionary mapping child_id -> list of parent_ids
    """
    edges = np.asarray(edges)
    forward_adj = defaultdict(list)
    reverse_adj = defaultdict(list)

    for parent, child in edges:
        forward_adj[int(parent)].append(int(child))
        reverse_adj[int(child)].append(int(parent))

    return dict(forward_adj), dict(reverse_adj)


def find_roots(edges: Union[np.ndarray, List[List[int]]]) -> np.ndarray:
    """
    Find all root nodes (nodes with no parents) in the DAG.

    Parameters
    ----------
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.

    Returns
    -------
    np.ndarray
        Array of root node IDs (nodes that appear as parents but never as children).
        Returns empty array if no edges are provided.
    """
    edges = np.asarray(edges)
    if len(edges) == 0:
        return np.array([], dtype=int)

    all_nodes = set(edges.flatten().astype(int))
    children = set(edges[:, 1].astype(int))
    return np.array(list(all_nodes - children), dtype=int)


def topological_sort(
    vertices: Union[np.ndarray, List[List[float]]],
    edges: Union[np.ndarray, List[List[int]]],
    cache: Optional[DAGCache] = None,
) -> np.ndarray:
    """
    Compute topological ordering of the DAG using depth-first search.

    Parameters
    ----------
    vertices : Union[np.ndarray, List[List[float]]]
        Array of shape (n_vertices, 3) containing 3D positions of vertices,
        or list of [x, y, z] coordinate lists.
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.
    cache : Optional[DAGCache], default=None
        Optional cache to store/retrieve the topological order for performance.

    Returns
    -------
    np.ndarray
        Array of vertex IDs in topological order. Parents appear before children.
        Isolated vertices are included at the end.
    """
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    if cache is not None and cache.topological_order is not None:
        return cache.topological_order

    forward_adj, _ = build_adjacency_lists(edges)
    roots = find_roots(edges)

    visited = set()
    result = []

    def dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        if node_id in forward_adj:
            for child in forward_adj[node_id]:
                dfs(child)
        result.append(node_id)

    # Start from all roots
    for root in roots:
        dfs(root)

    # Include any isolated vertices
    all_vertices = set(range(len(vertices)))
    for v in all_vertices - visited:
        result.append(v)

    result.reverse()
    result = np.array(result, dtype=int)
    if cache is not None:
        cache.topological_order = result

    return result


def compute_depths(
    vertices: Union[np.ndarray, List[List[float]]],
    edges: Union[np.ndarray, List[List[int]]],
    cache: Optional[DAGCache] = None,
) -> Dict[int, int]:
    """
    Compute depth of each node from root(s) using breadth-first search.

    Parameters
    ----------
    vertices : Union[np.ndarray, List[List[float]]]
        Array of shape (n_vertices, 3) containing 3D positions of vertices,
        or list of [x, y, z] coordinate lists.
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.
    cache : Optional[DAGCache], default=None
        Optional cache to store/retrieve the computed depths for performance.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping node_id to its depth from the nearest root node.
        Root nodes have depth 0, isolated vertices have depth 0.

    Notes
    -----
    Uses BFS to ensure shortest path distances from roots are computed.
    Multiple roots are supported - each subtree gets depths from its root.
    """
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    if cache is not None and cache.depths is not None:
        return cache.depths

    forward_adj, _ = build_adjacency_lists(edges)
    roots = find_roots(edges)

    depths = {}
    queue = deque(roots)

    # Initialize root depths
    for root in roots:
        depths[int(root)] = 0

    # BFS to compute depths
    while queue:
        current = queue.popleft()
        if current in forward_adj:
            for child in forward_adj[current]:
                if child not in depths:
                    depths[child] = depths[current] + 1
                    queue.append(child)

    # Set depth 0 for isolated vertices
    for v in range(len(vertices)):
        if v not in depths:
            depths[v] = 0

    if cache is not None:
        cache.depths = depths

    return depths


def build_parent_map(edges: Union[np.ndarray, List[List[int]]]) -> Dict[int, int]:
    """
    Build mapping from child to parent for tree traversal.

    Parameters
    ----------
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping child_id to parent_id. Nodes without parents
        (root nodes) will not appear as keys in this dictionary.

    Notes
    -----
    This assumes the graph is a forest (collection of trees). If a node
    has multiple parents, only the last one encountered will be stored.
    """
    edges = np.asarray(edges)
    parent_map = {}
    for parent, child in edges:
        parent_map[int(child)] = int(parent)
    return parent_map


def preprocess_lca(
    vertices: Union[np.ndarray, List[List[float]]],
    edges: Union[np.ndarray, List[List[int]]],
    cache: Optional[DAGCache] = None,
) -> Dict[int, Dict[int, int]]:
    """
    Preprocess DAG for O(log n) Lowest Common Ancestor queries using binary lifting.

    Parameters
    ----------
    vertices : Union[np.ndarray, List[List[float]]]
        Array of shape (n_vertices, 3) containing 3D positions of vertices,
        or list of [x, y, z] coordinate lists.
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.
    cache : Optional[DAGCache], default=None
        Optional cache to store/retrieve the binary lifting table for performance.

    Returns
    -------
    Dict[int, Dict[int, int]]
        Binary lifting table where ancestors[v][i] gives the 2^i-th ancestor of vertex v.
        Used for efficient LCA queries in O(log n) time complexity.

    Notes
    -----
    This preprocessing step creates a sparse table that allows LCA queries
    to be answered in O(log n) time. The space complexity is O(n log n).
    """
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    if cache is not None and cache.ancestors:
        return cache.ancestors

    depths = compute_depths(vertices, edges, cache)
    parent_map = build_parent_map(edges)

    max_depth = max(depths.values()) if depths else 0
    log_max = max_depth.bit_length() if max_depth > 0 else 1

    ancestors = defaultdict(dict)

    # Initialize direct parents
    for child, parent in parent_map.items():
        ancestors[child][0] = parent

    # Build binary lifting table
    for i in range(1, log_max):
        for node_id in range(len(vertices)):
            if i - 1 in ancestors[node_id]:
                prev_ancestor = ancestors[node_id][i - 1]
                if i - 1 in ancestors[prev_ancestor]:
                    ancestors[node_id][i] = ancestors[prev_ancestor][i - 1]

    result = dict(ancestors)

    if cache is not None:
        cache.ancestors = result

    return result


def lca(
    u: int,
    v: int,
    vertices: Union[np.ndarray, List[List[float]]],
    edges: Union[np.ndarray, List[List[int]]],
    cache: Optional[DAGCache] = None,
) -> Optional[int]:
    """
    Find Lowest Common Ancestor (LCA) of two nodes using binary lifting.

    Parameters
    ----------
    u : int
        First node ID.
    v : int
        Second node ID.
    vertices : Union[np.ndarray, List[List[float]]]
        Array of shape (n_vertices, 3) containing 3D positions of vertices,
        or list of [x, y, z] coordinate lists.
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.
    cache : Optional[DAGCache], default=None
        Optional cache containing preprocessed binary lifting table.

    Returns
    -------
    Optional[int]
        Node ID of the lowest common ancestor, or None if no common ancestor exists
        (nodes are in different connected components).

    Notes
    -----
    Requires preprocessing via preprocess_lca() for optimal O(log n) performance.
    If nodes are in different trees, returns None.
    """
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)

    # Handle trivial case
    if u == v:
        return u

    ancestors = preprocess_lca(vertices, edges, cache)
    depths = compute_depths(vertices, edges, cache)

    if u not in depths or v not in depths:
        return None

    # Make u the deeper node
    if depths[u] < depths[v]:
        u, v = v, u

    # Bring u to the same depth as v
    diff = depths[u] - depths[v]
    for i in range(diff.bit_length()):
        if diff & (1 << i):
            if i in ancestors.get(u, {}):
                u = ancestors[u][i]
            else:
                # If ancestor doesn't exist, nodes are in different components
                return None

    if u == v:
        return u

    # Binary search for LCA
    max_log = max(len(ancestors.get(u, {})), len(ancestors.get(v, {})))
    for i in range(max_log - 1, -1, -1):
        if i in ancestors.get(u, {}) and i in ancestors.get(v, {}):
            if ancestors[u][i] != ancestors[v][i]:
                u = ancestors[u][i]
                v = ancestors[v][i]

    # At this point, u and v are one step below their LCA
    # Handle root node case: if either node has no parent, it IS the root/LCA
    if 0 in ancestors.get(u, {}):
        return ancestors[u][0]
    elif 0 in ancestors.get(v, {}):
        return ancestors[v][0]
    else:
        # Both nodes are roots - they're the same root since u == v after depth adjustment
        return u


def shortest_path(
    source: int,
    target: int,
    csgraph: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Find shortest path between two nodes in the DAG via their LCA.

    Parameters
    ----------
    source : int
        Starting node ID.
    target : int
        Target node ID.
    vertices : Union[np.ndarray, List[List[float]]]
        Array of shape (n_vertices, 3) containing 3D positions of vertices,
        or list of [x, y, z] coordinate lists.
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.
    cache : Optional[DAGCache], default=None
        Optional cache for LCA preprocessing and path length storage.

    Returns
    -------
    Optional[np.ndarray]
        Array of node IDs forming the shortest path from source to target,
        or None if no path exists (nodes are in different components).

    Notes
    -----
    In a tree/DAG, there is exactly one path between any two connected nodes.
    The path goes: source -> ... -> LCA -> ... -> target.
    """
    d, P = sparse.csgraph.dijkstra(
        csgraph,
        indices=source,
        directed=False,
        unweighted=True,
        return_predecessors=True,
    )
    path_between = []
    if ~np.isinf(d[target]):
        current = target
        while current != source:
            path_between.append(current)
            current = P[current]
            if current is None:
                return None
        path_between.append(source)
        return np.array(path_between[::-1], dtype=int)
    else:
        return None


def path_length(
    source: int,
    target: int,
    vertices: Union[np.ndarray, List[List[float]]],
    edges: Union[np.ndarray, List[List[int]]],
    cache: Optional[DAGCache] = None,
) -> Optional[float]:
    """
    Compute Euclidean path length between two nodes along tree edges.

    Parameters
    ----------
    source : int
        Starting node ID.
    target : int
        Target node ID.
    vertices : Union[np.ndarray, List[List[float]]]
        Array of shape (n_vertices, 3) containing 3D positions of vertices,
        or list of [x, y, z] coordinate lists.
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.
    cache : Optional[DAGCache], default=None
        Optional cache for storing computed path lengths to avoid recomputation.

    Returns
    -------
    Optional[float]
        Sum of Euclidean distances between consecutive vertices along the path,
        or None if no path exists between the nodes.

    Notes
    -----
    Uses caching to avoid recomputing the same path lengths repeatedly.
    Path lengths are symmetric: path_length(u, v) == path_length(v, u).
    """
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    # Check cache
    if cache is not None and cache.path_lengths:
        cache_key = (min(source, target), max(source, target))
        if cache_key in cache.path_lengths:
            return cache.path_lengths[cache_key]

    path = shortest_path(source, target, vertices, edges, cache)
    if path is None:
        return None

    # Compute Euclidean length
    length = single_path_length(path, vertices, edges)

    # Store in cache
    if cache is not None:
        if cache.path_lengths is None:
            cache.path_lengths = {}
        cache_key = (min(source, target), max(source, target))
        cache.path_lengths[cache_key] = length

    return length


def minimum_spanning_tree_points(
    point_ids: Union[np.ndarray, List[int]],
    vertices: Union[np.ndarray, List[List[float]]],
    edges: Union[np.ndarray, List[List[int]]],
    cache: Optional[DAGCache] = None,
) -> np.ndarray:
    """
    Find minimum spanning tree connecting a set of points in the DAG.

    Parameters
    ----------
    point_ids : Union[np.ndarray, List[int]]
        Array or list of vertex IDs that need to be connected.
    vertices : Union[np.ndarray, List[List[float]]]
        Array of shape (n_vertices, 3) containing 3D positions of vertices,
        or list of [x, y, z] coordinate lists.
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.
    cache : Optional[DAGCache], default=None
        Optional cache for LCA preprocessing to speed up computations.

    Returns
    -------
    np.ndarray
        Array of edges (parent, child) forming the minimum spanning tree
        that connects all specified points with minimal total edge count.

    Notes
    -----
    Uses Steiner tree approach: finds all LCAs between point pairs,
    then extracts the minimal subtree connecting all points and LCAs.
    """
    point_ids = np.asarray(point_ids)
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    if len(point_ids) == 0:
        return np.array([], dtype=int).reshape(0, 2)

    if len(point_ids) == 1:
        return np.array([], dtype=int).reshape(0, 2)

    # Find all pairwise LCAs
    lcas = set(point_ids)
    for i in range(len(point_ids)):
        for j in range(i + 1, len(point_ids)):
            lca_node = lca(point_ids[i], point_ids[j], vertices, edges, cache)
            if lca_node is not None:
                lcas.add(lca_node)

    # Find the subtree induced by these nodes
    parent_map = build_parent_map(edges)
    result_edges = []

    for node_id in lcas:
        if node_id in parent_map and parent_map[node_id] in lcas:
            result_edges.append((parent_map[node_id], node_id))

    return np.array(result_edges, dtype=int)


def get_subtree_nodes(subtree_root: int, edges: Union[np.ndarray, list]) -> np.ndarray:
    """
    Get all nodes in the subtree rooted at given node.

    Parameters
    ----------
    subtree_root : int
        Root node ID of subtree to traverse.
    edges : Union[np.ndarray, list]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id].

    Returns
    -------
    np.ndarray
        Array of all node IDs in the subtree, including the root.

    Notes
    -----
    Uses breadth-first search to traverse the subtree from the given root node.
    """
    edges = np.asarray(edges)
    forward_adj, _ = build_adjacency_lists(edges)
    subtree = set()
    queue = deque([subtree_root])

    while queue:
        current = queue.popleft()
        subtree.add(current)
        if current in forward_adj:
            queue.extend(forward_adj[current])

    return np.array(list(subtree), dtype=int)


def find_end_points(
    csgraph,
) -> np.ndarray:
    return np.flatnonzero(csgraph.sum(axis=0).flatten() == 0)


def find_branch_points(
    csgraph,
) -> np.ndarray:
    return np.flatnonzero(csgraph.sum(axis=0).flatten() > 1)


def find_leaf_nodes(
    vertices: Union[np.ndarray, List[List[float]]],
    edges: Union[np.ndarray, List[List[int]]],
    cache: Optional[DAGCache] = None,
) -> np.ndarray:
    """
    Find all leaf nodes (nodes with no children) in the DAG.

    Parameters
    ----------
    vertices : Union[np.ndarray, List[List[float]]]
        Array of shape (n_vertices, 3) containing 3D positions of vertices,
        or list of [x, y, z] coordinate lists.
    edges : Union[np.ndarray, List[List[int]]]
        Array of shape (n_edges, 2) where each row is [parent_id, child_id],
        or list of [parent_id, child_id] pairs.

    Returns
    -------
    np.ndarray
        Array of node IDs that have no children (leaf nodes).
        Includes isolated vertices that don't appear in any edges.

    Notes
    -----
    Leaf nodes represent endpoints of the tree structure and are often
    important for analysis of tree topology and traversal algorithms.
    """
    if cache is not None:
        if cache.branch_points is not None:
            return cache.branch_points

    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    forward_adj, _ = build_adjacency_lists(edges)
    all_nodes = set(range(len(vertices)))
    leaf_nodes = [
        node_id
        for node_id in all_nodes
        if node_id not in forward_adj or len(forward_adj[node_id]) == 0
    ]
    if cache is not None:
        cache.end_points = np.array(leaf_nodes, dtype=int)
    return np.array(leaf_nodes, dtype=int)


def source_target_distances(
    sources: Union[np.ndarray, list],
    targets: Union[np.ndarray, list],
    csgraph: Optional[sparse.csgraph],
    limit: Optional[float] = None,
) -> np.ndarray:
    """
    Compute shortest path distances between specific sources and targets.

    Parameters
    ----------
    sources : Union[np.ndarray, list]
        Array or list of source node IDs.
    targets : Union[np.ndarray, list]
        Array or list of target node IDs.
    csgraph : Optional[sparse.csgraph]
        Compressed sparse graph representation of the mesh.

    Returns
    -------
    np.ndarray
        Array of shortest path distances between each source and target.
    """
    sources = np.asarray(sources)
    targets = np.asarray(targets)
    if limit is None:
        limit = np.inf

    do_transpose = False
    if len(sources) > len(targets):
        sources, targets = targets, sources
        do_transpose = True
    distances = sparse.csgraph.dijkstra(
        csgraph,
        directed=False,
        indices=sources,
        return_predecessors=False,
        limit=limit,
    )
    if do_transpose:
        return distances[:, targets].T
    return distances[:, targets]


def cut_graph(vertices, edges, cut_inds, directed=True, euclidean_weight=True):
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
        Graph with vertices in vinds cut off from parents.
    """
    e_keep = ~np.isin(edges[:, 0], cut_inds)
    es_new = edges[e_keep]
    return build_csgraph(
        vertices, es_new, euclidean_weight=euclidean_weight, directed=directed
    )


def build_segments(
    vertices,
    edges,
    branch_points,
    child_nodes,
    hops_to_root,
):
    segments = []
    segment_map = np.full(len(vertices), -1)
    if len(branch_points) > 0:
        ch_inds = np.concatenate([child_nodes[bp] for bp in branch_points])
    else:
        ch_inds = np.array([], dtype=int)
    g = cut_graph(vertices, edges, ch_inds)
    _, ls = sparse.csgraph.connected_components(g)
    _, invs = np.unique(ls, return_inverse=True)
    for ii in np.unique(invs):
        seg = np.flatnonzero(invs == ii)
        segments.append(seg[np.argsort(hops_to_root[seg])[::-1]])
    segment_map = invs
    return segments, segment_map


def build_cover_paths(
    end_points: np.ndarray,
    parent_node_array: np.ndarray,
    distance_to_root: np.ndarray,
    cache: Optional[DAGCache] = None,
    include_parent: bool = False,
) -> list[list[int]]:
    cover_paths = []
    seen = np.full(len(parent_node_array), False, dtype=bool)
    end_points = end_points[np.argsort(distance_to_root[end_points])][
        ::-1
    ]  # sort in order farthest to closest.
    for ep in end_points:
        path = []
        current = ep
        while not seen[current]:
            seen[current] = True
            path.append(int(current))
            current = parent_node_array[current]
        if include_parent and current != -1:
            path.append(int(current))
        cover_paths.append(np.array(path, dtype=int))
    if cache is not None:
        cache.cover_paths = cover_paths
    return cover_paths


def build_proximity_lists_chunked(
    vertices, csgraph, distance_threshold, chunk_size=1000
):
    n_vertices = len(vertices)
    index_list = []
    proximity_list = []
    for start_idx in range(0, n_vertices, chunk_size):
        end_idx = min(start_idx + chunk_size, n_vertices)
        indices = np.arange(start_idx, end_idx)

        distances = sparse.csgraph.dijkstra(
            csgraph, indices=indices, directed=False, limit=distance_threshold
        )
        for ii, idx in enumerate(indices):
            proximal_indices = np.flatnonzero(distances[ii] <= distance_threshold)
            index_list.append([idx] * len(proximal_indices))
            proximity_list.append(proximal_indices.tolist())

    return np.concatenate(index_list), np.concatenate(proximity_list)
