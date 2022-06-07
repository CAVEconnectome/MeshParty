from .utils import decompress_mesh_data
from ..skeleton import Skeleton
from ..trimesh_io import Mesh
import h5py
import os
import orjson
import pandas as pd
import warnings
import numpy as np
from dataclasses import asdict

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

NULL_VERSION = 1
LATEST_VERSION = 2


def save_meshwork_metadata(filename, mw, version=LATEST_VERSION):
    with h5py.File(filename, "a") as f:
        f.attrs["voxel_resolution"] = mw.anno.voxel_resolution
        f.attrs["version"] = version
        if mw.seg_id is not None:
            f.attrs["seg_id"] = mw.seg_id


def load_meshwork_metadata(filename):
    meta = {}
    with h5py.File(filename, "r") as f:
        meta["seg_id"] = f.attrs.get("seg_id", None)
        meta["voxel_resolution"] = f.attrs.get("voxel_resolution", None)
        meta["version"] = f.attrs.get("version", NULL_VERSION)
    return meta


def save_meshwork_mesh(filename, mw, version=LATEST_VERSION):
    mesh_mask = mw.mesh_mask
    if mw._original_mesh_data is not None:
        vs, fs, es, nm, vxsc = decompress_mesh_data(*mw._original_mesh_data)
        mesh = Mesh(vs, fs, link_edges=es, node_mask=nm, voxel_scaling=vxsc)
    else:
        mesh = mw.mesh

    with h5py.File(filename, "a") as f:
        f.create_group("mesh")
        f.create_dataset("mesh/vertices", data=mesh.vertices, compression="gzip")
        f.create_dataset("mesh/faces", data=mesh.faces, compression="gzip")
        f.create_dataset("mesh/node_mask", data=mesh.node_mask, compression="gzip")
        if mesh.voxel_scaling is not None:
            f["mesh"].attrs["voxel_scaling"] = mesh.voxel_scaling
        if mesh.link_edges is not None:
            f.create_dataset(
                "mesh/link_edges", data=mesh.link_edges, compression="gzip"
            )
        f.create_dataset("mesh/mesh_mask", data=mesh_mask, compression="gzip")


def load_meshwork_mesh(filename, version=NULL_VERSION):
    with h5py.File(filename, "r") as f:
        verts = f["mesh/vertices"][()]
        faces = f["mesh/faces"][()]
        if len(faces.shape) == 1:
            faces = faces.reshape(-1, 3)

        if "link_edges" in f["mesh"].keys():
            link_edges = f["mesh/link_edges"][()]
        else:
            link_edges = None

        node_mask = f["mesh/node_mask"][()]
        voxel_scaling = f["mesh"].attrs.get("voxel_scaling", None)
        mesh_mask = f["mesh/mesh_mask"][()]
    return (
        Mesh(
            vertices=verts,
            faces=faces,
            link_edges=link_edges,
            node_mask=node_mask,
            voxel_scaling=voxel_scaling,
        ),
        mesh_mask,
    )


def save_meshwork_skeleton(filename, mw, version=LATEST_VERSION):
    if mw.skeleton is None:
        return

    sk = mw.skeleton.reset_mask()
    with h5py.File(filename, "a") as f:
        f.create_group("skeleton")
        f.create_dataset("skeleton/vertices", data=sk.vertices, compression="gzip")
        f.create_dataset("skeleton/edges", data=sk.edges, compression="gzip")
        f.create_dataset("skeleton/root", data=sk.root)
        f.create_dataset(
            "skeleton/meta",
            data=np.string_(
                orjson.dumps(asdict(sk.meta), option=orjson.OPT_SERIALIZE_NUMPY)
            ),
        )

        f.create_dataset(
            "skeleton/mesh_to_skel_map", data=sk.mesh_to_skel_map, compression="gzip"
        )
        if sk.radius is not None:
            f.create_dataset("skeleton/radius", data=sk.radius, compression="gzip")
        if sk.mesh_index is not None:
            f.create_dataset(
                "skeleton/mesh_index", data=sk.mesh_index, compression="gzip"
            )
        if sk.voxel_scaling is not None:
            f["skeleton"].attrs["voxel_scaling"] = sk.voxel_scaling


def load_meshwork_skeleton(filename, version=NULL_VERSION):
    with h5py.File(filename, "r") as f:
        if "skeleton" not in f:
            return None

        verts = f["skeleton/vertices"][()]
        edges = f["skeleton/edges"][()]
        root = f["skeleton/root"][()]
        mesh_to_skel_map = f["skeleton/mesh_to_skel_map"][()]
        if "radius" in f["skeleton"].keys():
            radius = f["skeleton/radius"][()]
        else:
            radius = None
        voxel_scaling = f["skeleton"].attrs.get("voxel_scaling", None)

        if "meta" in f["skeleton"].keys():
            meta = orjson.loads(f["skeleton/meta"][()].tobytes())
        else:
            meta = {}

        if "mesh_index" in f["skeleton"].keys():
            mesh_index = f["skeleton/mesh_index"][()]
        else:
            mesh_index = None
        return Skeleton(
            verts,
            edges,
            root=root,
            radius=radius,
            mesh_to_skel_map=mesh_to_skel_map,
            mesh_index=mesh_index,
            voxel_scaling=voxel_scaling,
            meta=meta,
        )


def _save_dataframe_generic(df, table_name, filename):
    key = f"annotations/{table_name}/data"
    dat = orjson.dumps(
        df.to_dict(), option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
    )
    with h5py.File(filename, "a") as f:
        f.create_dataset(key, data=np.string_(dat))
    pass


def _load_dataframe_generic(filename, table_name):
    key = f"annotations/{table_name}/data"
    with h5py.File(filename, "r") as f:
        dat = f[key][()].tobytes()
        df = pd.DataFrame.from_records(orjson.loads(dat))
        try:
            df.index = np.array(df.index, dtype="int")
        except:
            pass
    return df


def _save_dataframe_pandas(data, table_name, filename):
    data.to_hdf(
        filename, f"annotations/{table_name}/data", complib="blosc", complevel=5
    )
    pass


def _load_dataframe_pandas(filename, table_name):
    return pd.read_hdf(filename, f"annotations/{table_name}/data")


anno_save_function = {1: _save_dataframe_pandas, 2: _save_dataframe_generic}
anno_load_function = {1: _load_dataframe_pandas, 2: _load_dataframe_generic}


def save_meshwork_annotations(filename, mw, version=LATEST_VERSION):
    annos = mw.anno
    if version not in anno_save_function:
        raise ValueError(f"Version must be one of {list(anno_save_function)}")

    for table_name in annos.table_names:
        with h5py.File(filename, "a") as f:
            dset = f.create_group(f"annotations/{table_name}")
            anno = annos[table_name]
            dset.attrs["anchor_to_mesh"] = int(anno.anchored)
            if anno.point_column is not None:
                dset.attrs["point_column"] = anno.point_column
            dset.attrs["max_distance"] = anno._max_distance
            dset.attrs["defined_index"] = int(anno._defined_index)
            if anno._defined_index is True:
                dset.attrs["index_column"] = anno.index_column_original

        anno_save_function[version](
            annos[table_name].data_original, table_name, filename
        )


def load_meshwork_annotations(filename, version=NULL_VERSION):

    with h5py.File(filename, "r") as f:
        if "annotations" not in f:
            return {}
        table_names = list(f["annotations"].keys())

    annotation_dfs = {}
    for table_name in table_names:
        annotation_dfs[table_name] = {}
        annotation_dfs[table_name]["data"] = anno_load_function[version](
            filename, table_name
        )
        with h5py.File(filename, "r") as f:
            dset = f[f"annotations/{table_name}"]
            annotation_dfs[table_name]["anchor_to_mesh"] = bool(
                dset.attrs.get("anchor_to_mesh")
            )
            annotation_dfs[table_name]["point_column"] = dset.attrs.get(
                "point_column", None
            )
            annotation_dfs[table_name]["max_distance"] = dset.attrs.get("max_distance")
            if bool(dset.attrs.get("defined_index", False)):
                annotation_dfs[table_name]["index_column"] = dset.attrs.get(
                    "index_column", None
                )
    return annotation_dfs


def _save_meshwork(filename, mw, overwrite=False, version=LATEST_VERSION):
    if isinstance(filename, str):
        if os.path.exists(filename):
            if overwrite is False:
                raise FileExistsError()
            else:
                print(f"\tDeleting existing data in {filename}...")
                with h5py.File(filename, "w") as f:
                    pass

    save_meshwork_metadata(filename, mw, version=version)
    save_meshwork_mesh(filename, mw, version=version)
    save_meshwork_skeleton(filename, mw, version=version)
    save_meshwork_annotations(filename, mw, version=version)


def _load_meshwork(filename):
    meta = load_meshwork_metadata(filename)
    version = meta.get("version")
    mesh, mask = load_meshwork_mesh(filename, version=version)
    skel = load_meshwork_skeleton(filename, version=version)
    annos = load_meshwork_annotations(filename, version=version)
    return meta, mesh, skel, annos, mask
