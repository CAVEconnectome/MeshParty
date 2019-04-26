import os
from . import trimesh_io
import h5py
from meshparty import skeleton


def write_skeleton_h5(sk, filename, overwrite=False):
    '''
    Write a skeleton and its properties to an hdf5 file.

    :param sk: Skeleton
    :param filename: String. Filename of skeleton file.
    :param overwrite: Boolean, (default False). Allows overwriting.
    '''
    write_skeleton_h5_by_part(filename, sk.vertices, sk.edges,
                              sk.vertex_properties, sk.edge_properties,
                              sk.root, overwrite=overwrite)


def write_skeleton_h5_by_part(filename, vertices, edges, vertex_properties={},
                              edge_properties={}, root=None, overwrite=False):
    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
        else:
            return
    with h5py.File(filename, 'w') as f:
        f.create_dataset('vertices', data=vertices, compression='gzip')
        f.create_dataset('edges', data=edges, compression='gzip')
        if len(vertex_properties) > 0:
            _write_dict_to_group(f, 'vertex_properties', vertex_properties)
        if len(edge_properties) > 0:
            _write_dict_to_group(f, 'edge_properties', edge_properties)
        if root is not None:
            f.create_dataset('root', data=root, compression='gzip')


def _write_dict_to_group(f, group_name, data_dict):
    d_grp = f.create_group(group_name)
    for d_name, d_data in data_dict.items():
        d_grp.create_dataset(d_name, data=d_data)


def read_skeleton_h5_by_part(filename):
    assert os.path.isfile(filename)

    with h5py.File(filename, 'r') as f:
        vertices = f['vertices'].value
        edges = f['edges'].value

    vertex_properties = {}
    if 'vertex_properties' in f.keys():
        for vp_key in f['vertex_properties'].keys():
            vertex_properties[vp_key] = f['vertex_properties'][vp_key].value

    edge_properties = {}
    if 'edge_properties' in f.keys():
        for ep_key in f['edge_properties'].keys():
            edge_properties[ep_key] = f['edge_properties'][ep_key].value

    # vertex_lists = {}
    # if 'vertex_lists' in f.keys():
    #     for vl_key in f['vertex_lists'].keys():
    #         vertex_lists[vl_key] = f['vertex_lists'][vl_key].value

    if 'root' in f.keys():
        root = f['root'].value
    else:
        root = None
    return vertices, edges, vertex_properties, edge_properties, root


def read_skeleton_h5(filename):
    '''
    Reads a skeleton and its properties from an hdf5 file.

    :param filename: String. Filename of skeleton file.
    '''
    vertices, edges, vertex_properties, edge_properties, root = read_skeleton_h5_by_part(
        filename)
    return skeleton.SkeletonForest(vertices=vertices,
                                   edges=edges,
                                   vertex_properties=vertex_properties,
                                   edge_properties=edge_properties,
                                   root=root)


def export_to_swc(skel, filename, node_labels=None, radius=None, header=None, xyz_scaling=1000):
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

    if header is None:
        header_string = ''
    else:
        header_string = '\n'.join(['{}: {}'.format(k, v)
                                   for k, v in header.items()])

    if radius is None:
        radius = np.full(len(skel.vertices), 1000)
    elif np.issubdtype(type(radius), int):
        radius = np.full(len(skel.vertices), radius)

    if node_labels is None:
        node_labels = np.full(len(skel.vertices), 3)

    swc_dat = _build_swc_array(skel, node_labels, radius, xyz_scaling)

    with open(filename, 'w') as f:
        np.savetxt(f, swc_dat, delimiter=' ', header=header_string, comments='#',
                   fmt=['%i', '%i', '%.3f', '%.3f', '%.3f', '%.3f', '%i'])


def _build_swc_array(skel, node_labels, radius, xyz_scaling):
    '''
    Helper function for producing the numpy table for an swc.
    '''
    ds = skel.distance_to_root
    order_old = np.argsort(ds)
    new_ids = np.arange(len(ds))
    order_map = dict(zip(order_old, new_ids))

    node_labels = node_labels[order_old]
    xyz = skel.vertices[order_old]
    radius = radius[order_old]
    par_ids = np.array([order_map.get(nid, -1)
                        for nid in skel._parent_node_array[order_old]])

    swc_dat = np.hstack((new_ids[:, np.newaxis],
                         node_labels[:, np.newaxis],
                         xyz / xyz_scaling,
                         radius[:, np.newaxis] / xyz_scaling,
                         par_ids[:, np.newaxis]))
    return swc_dat
