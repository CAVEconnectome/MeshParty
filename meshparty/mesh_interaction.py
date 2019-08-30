from sklearn import cluster
from scipy import sparse

def mesh_mesh_proximity(target_mesh, query_mesh, max_dist=75):
    '''
    Finds the vertices on a target mesh that are close to points on a query mesh.
    '''
    proximate_point_list = target_mesh.kdtree.query_ball_tree(query_mesh.kdtree, max_dist)
    is_proximate = np.array(list(map(len, proximate_point_list)))>0
    return is_proximate


def mesh_mesh_contacts(target_mesh,
                       query_mesh,
                       max_dist=75,
                       cluster_min_size=10,
                       cluster_eps=250):
    '''
    :param target_mesh: Mesh for which we return contact indices and do vertex clustering
    :param other_mesh: Mesh for which we search for contacts.
    :param max_dist: Distance (in euclidean space) for which to consider contact between mesh vertices.
    :param cluster_min_size: Minimum size of a contact point cluster. Smaller outliers are ignored.
    :param cluster_eps: Distance (along mesh graph) for the max distance for two points to be located and
                        be considered within the same neighborhood for clustering. See sklearn.cluster.DBSCAN
    :returns: Array with row for each vertex in target mesh, with 0 for no contact and unique nonzero integers
              for each distinct contact.
              
    '''
    is_proximate = mesh_mesh_proximity(target_mesh, query_mesh, max_dist)
    proximate_inds = np.flatnonzero(is_proximate)
    ds_long = sparse.csgraph.dijkstra(target_mesh.csgraph, indices=proximate_inds)
    ds = ds_long[:, proximate_inds]

    dbscan = cluster.DBSCAN(eps=cluster_eps,
                            metric='precomputed',
                            min_samples=cluster_min_size)
    clust_res = dbscan.fit(ds)
    vert_labels = is_proximate.astype(int)
    vert_labels[proximate_inds] = clust_res.labels_ + 1
    return vert_labels
