import os
import h5py
import orjson
import json

from .utils_io import NumpyEncoder
import numpy as np
from meshparty import skeleton
from dataclasses import asdict

FILE_VERSION = 2


def write_skeleton_h5(sk, filename, overwrite=False):
    """
    Write a skeleton and its properties to an hdf5 file.

    Parameters
    ----------
    sk : :obj:`meshparty.skeleton.Skeleton`
        new_mesh_filt
    filename : str
        Filename of skeleton file.
    overwrite : bool
         Allows overwriting.(default False).
    """
    if not hasattr(sk, "meta"):
        sk.meta = {}

    write_skeleton_h5_by_part(
        filename,
        vertices=sk.vertices,
        edges=sk.edges,
        meta=sk.meta,
        mesh_to_skel_map=sk.mesh_to_skel_map,
        vertex_properties=sk.vertex_properties,
        root=sk.root,
        overwrite=overwrite,
    )


def write_skeleton_h5_by_part(
    filename,
    vertices,
    edges,
    meta,
    mesh_to_skel_map=None,
    vertex_properties={},
    root=None,
    overwrite=False,
):
    """
    Helper function for writing all parts of a skeleton file to an h5.

    Parameters
    ----------
    filename : str
        path to write
    vertices : np.array
        Nx3 numpy array of vertex locations
    edges : np.array
        Kx2 numpy array of vertex indices for edges
    mesh_to_skel_map : np.array
        M long numpy array.  M is the number of vertices in a mesh that this
        is associated with.  The entries are indices into the N skeleton vertices
    vertex_properties : dict
        a dictionary of np.arrays, were keys are descriptive labels
        and the values are arrays that quantify that label at each vertex
        examples..
        mesh_index) the mesh index of this vertex
        rs) the sdf (local caliber/thickness) of the mesh at each index
    root : int
        which vertex index is root
    overwrite : bool
        whether to overwrite file

    """

    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
        else:
            return
    with h5py.File(filename, "w") as f:
        f.attrs["file_version"] = FILE_VERSION

        f.create_dataset("vertices", data=vertices, compression="gzip")
        f.create_dataset("edges", data=edges, compression="gzip")
        f.create_dataset(
            "meta",
            data=np.string_(
                orjson.dumps(asdict(meta), option=orjson.OPT_SERIALIZE_NUMPY)
            ),
        )
        if mesh_to_skel_map is not None:
            f.create_dataset(
                "mesh_to_skel_map", data=mesh_to_skel_map, compression="gzip"
            )
        if len(vertex_properties) > 0:
            _write_dict_to_group(f, "vertex_properties", vertex_properties)
        if root is not None:
            f.create_dataset("root", data=root)


def _write_dict_to_group(f, group_name, data_dict):
    d_grp = f.create_group(group_name)
    for d_name, d_data in data_dict.items():
        d_grp.create_dataset(d_name, data=json.dumps(d_data, cls=NumpyEncoder))


def read_skeleton_h5_by_part(filename):
    """
    Helper function for extracting all parts of a skeleton file from an h5.

    Parameters
    ----------
    filename : str
        path to a h5 file with skeletons

    Returns
    -------
    str
        filename, path to write
    np.array
        vertices, Nx3 numpy array of vertex locations
    np.array
        edges , Kx2 numpy array of vertex indices for edges
    np.array
        mesh_to_skel_map , M long numpy array.  M is the number of vertices in a mesh that this
        is associated with.  The entries are indices into the N skeleton vertices
    dict
        vertex_properties, a dictionary of np.arrays, were keys are descriptive labels
        and the values are arrays that quantify that label at each vertex
        examples..
        mesh_index) the mesh index of this vertex
        rs) the sdf (local caliber/thickness) of the mesh at each index
    int
        root, which vertex index is root
    bool
        overwrite, whether to overwrite file

    """
    assert os.path.isfile(filename)

    with h5py.File(filename, "r") as f:
        vertices = f["vertices"][()]
        edges = f["edges"][()]

        if "mesh_to_skel_map" in f.keys():
            mesh_to_skel_map = f["mesh_to_skel_map"][()]
        else:
            mesh_to_skel_map = None

        vertex_properties = {}
        if "vertex_properties" in f.keys():
            for vp_key in f["vertex_properties"].keys():
                vertex_properties[vp_key] = json.loads(
                    f["vertex_properties"][vp_key][()], object_hook=_convert_keys_to_int
                )

        if "meta" in f.keys():
            dat = f["meta"][()].tobytes()
            meta = orjson.loads(dat)
        else:
            meta = {}

        if "root" in f.keys():
            root = f["root"][()]
        else:
            root = None

    return vertices, edges, meta, mesh_to_skel_map, vertex_properties, root


def read_skeleton_h5(filename, remove_zero_length_edges=False):
    """
    Reads a skeleton and its properties from an hdf5 file.

    Parameters
    ----------
    filename: str
        path to skeleton file
    remove_zero_length_edges: bool, optional
        If True, post-processes the skeleton data to removes any zero
        length edges. Default is False.
    Returns
    -------
    :obj:`meshparty.skeleton.Skeleton`
        skeleton object loaded from the h5 file

    """
    (
        vertices,
        edges,
        meta,
        mesh_to_skel_map,
        vertex_properties,
        root,
    ) = read_skeleton_h5_by_part(filename)
    return skeleton.Skeleton(
        vertices=vertices,
        edges=edges,
        mesh_to_skel_map=mesh_to_skel_map,
        vertex_properties=vertex_properties,
        root=root,
        remove_zero_length_edges=remove_zero_length_edges,
        meta=meta,
    )


def export_to_swc(
    skel, filename, node_labels=None, radius=None, header=None, xyz_scaling=1000
):
    """
    Export a skeleton file to an swc file
    (see http://research.mssm.edu/cnic/swc.html for swc definition)

    Parameters
    ----------
    filename : str
        path to the file to save the swc to
    node_labels : iterable
        None (default) or an interable of ints co-indexed with vertices.
        Corresponds to the swc node categories. Defaults to setting all
        nodes to label 3, dendrite.
    radius : iterable
        None (default) or an iterable of floats. This should be co-indexed with vertices.
        Radius values are assumed to be in the same units as the node vertices.
    header : str or list, default None.
        An optional header string for the file. Each element of the list
    xyz_scaling: Number, default 1000. Down-scales spatial units from the skeleton's units to
                        whatever is desired by the swc. E.g. nm to microns has scaling=1000.
    """

    if header is None:
        header_string = ""
    else:
        if isinstance(header, str):
            header = [header]
        header[0] = " ".join(["#", header[0]])
        header_string = "\n# ".join(header)

    if radius is None:
        radius = np.full(len(skel.vertices), 1000)
    elif np.issubdtype(type(radius), int):
        radius = np.full(len(skel.vertices), radius)

    if node_labels is None:
        node_labels = np.full(len(skel.vertices), 3)

    swc_dat = _build_swc_array(skel, node_labels, radius, xyz_scaling)

    with open(filename, "w") as f:
        np.savetxt(
            f,
            swc_dat,
            delimiter=" ",
            header=header_string,
            comments="#",
            fmt=["%i", "%i", "%.3f", "%.3f", "%.3f", "%.3f", "%i"],
        )


def _build_swc_array(skel, node_labels, radius, xyz_scaling):
    """
    Helper function for producing the numpy table for an swc.
    """
    order_old = np.concatenate([p[::-1] for p in skel.cover_paths])
    new_ids = np.arange(skel.n_vertices)
    order_map = dict(zip(order_old, new_ids))

    node_labels = np.array(node_labels)[order_old]
    xyz = skel.vertices[order_old]
    radius = radius[order_old]
    par_ids = np.array([order_map.get(nid, -1) for nid in skel.parent_nodes(order_old)])

    swc_dat = np.hstack(
        (
            new_ids[:, np.newaxis],
            node_labels[:, np.newaxis],
            xyz / xyz_scaling,
            radius[:, np.newaxis] / xyz_scaling,
            par_ids[:, np.newaxis],
        )
    )
    return swc_dat


def _convert_keys_to_int(x):
    if type(x) is dict:
        return {int(k): v for k, v in x.items()}
    else:
        return x
