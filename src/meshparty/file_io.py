import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, BinaryIO
from urllib import parse

import cloudfiles
import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
from numpy import savez_compressed
from scipy.sparse import load_npz, save_npz

from .base import MeshWorkSync
from .data_layers import GraphSync, Link, PointCloudSync, SkeletonSync

if TYPE_CHECKING:
    import scipy

__all__ = ["load_meshwork", "save_meshwork", "MeshworkIO"]

PUTABLE_SCHEMES = ["s3", "gs", "file", "mem"]
METADATA_FILENAME = "metadata.json"


def load_meshwork(source: Union[str, BinaryIO]) -> MeshWorkSync:
    """Load a MeshWorkSync object from a file path or file object.

    Parameters
    ----------
    source : Union[str, BinaryIO]
        The path to the file/cloudpath or an open binary file object.

    Returns
    -------
    MeshWorkSync
        The loaded MeshWorkSync object.

    """
    if isinstance(source, str):
        return _load_from_path(source)
    else:
        return _load_from_file_object(source)


def _load_from_path(path: str) -> MeshWorkSync:
    """Load from a file path using cloudfiles."""
    filename = path.split("/")[-1]
    basepath = "/".join(path.split("/")[:-1])
    return MeshworkIO(basepath).load(filename)


def _load_from_file_object(file_obj: BinaryIO) -> MeshWorkSync:
    """Load from an open binary file object."""
    file_bytes = file_obj.read()
    return _import_neuron_from_bytes(file_bytes)


def save_meshwork(
    nrn: MeshWorkSync,
    file: Union[str, BinaryIO, None] = None,
    allow_overwrite: bool = False,
) -> None:
    """Save MeshWorkSync to a file path or file object.

    Parameters
    ----------
    nrn : MeshWorkSync
        The MeshWorkSync object to save.
    file: Union[str, BinaryIO, None]
        File path, open binary file object, or None (uses nrn.name).
    allow_overwrite : bool
        Whether to allow overwriting existing files.
    """
    # Handle deprecated file parameter
    if isinstance(file, str) or file is None:
        _save_to_path(nrn, file, allow_overwrite)
    else:
        _save_to_file_object(nrn, file)


def _save_to_path(
    nrn: MeshWorkSync, path: Optional[str], allow_overwrite: bool
) -> None:
    """Save to a file path using cloudfiles."""
    if path:
        basepath = "/".join(path.split("/")[:-1])
        name = path.split("/")[-1]
    else:
        basepath = "."
        name = f"{nrn.name}.mwk"
    MeshworkIO(basepath).save(nrn, name, allow_overwrite=allow_overwrite)


def _save_to_file_object(nrn: MeshWorkSync, file_obj: BinaryIO) -> None:
    """Save to an open binary file object."""
    tar_bytes = _export_neuron_to_bytes(nrn)
    file_obj.write(tar_bytes)


def _export_neuron_to_bytes(nrn: MeshWorkSync) -> bytes:
    """Export MeshWorkSync to bytes data."""
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as tf:
        _export_neuron_to_tar(nrn, tf)
    return b.getvalue()


def _export_neuron_to_tar(nrn: MeshWorkSync, tf: tarfile.TarFile) -> None:
    """Export MeshWorkSync to tar file."""
    # export metadata
    add_file_to_tar(
        name=METADATA_FILENAME,
        data=extract_metadata(nrn),
        tf=tf,
    )

    # Export all layers
    for l in nrn.layers:
        match l.layer_type:
            case "skeleton":
                export_skeleton_layer(l, tf)
            case "graph":
                export_graph_layer(l, tf)
            case "mesh":
                export_mesh_layer(l, tf)
            case "points":
                export_point_cloud_layer(l, tf, as_annotation=False)

    # Export all annotations
    for anno in nrn.annotations:
        export_point_cloud_layer(anno, tf, as_annotation=True)

    # Export linkage
    export_linkage(nrn, tf)


def _import_neuron_from_bytes(file: bytes) -> MeshWorkSync:
    """Import MeshWorkSync from bytes data."""
    with tarfile.open(fileobj=io.BytesIO(file), mode="r") as tf:
        files = {f.name: f for f in tf.getmembers()}
        metadata = load_dict(files[METADATA_FILENAME], tf)
        _validate_archive_structure(metadata, files)

        nrn = MeshWorkSync(name=metadata["name"], meta=metadata["meta"])
        nrn = _process_layers(metadata, files, tf, nrn)
        nrn = _process_annotations(metadata, files, tf, nrn)
        nrn = _process_linkage(metadata, tf, nrn)
    return nrn


def _validate_archive_structure(metadata: dict, files: dict):
    """Validate that all expected files exist in the archive."""
    missing_files = []

    # Check required files exist
    for layer_name, layer_info in metadata["structure"]["layers"].items():
        required_files = [f"layers/{layer_name}/meta.json"]

        # Add type-specific required files
        if layer_info["type"] in ["skeleton", "graph"]:
            required_files.extend(
                [
                    f"layers/{layer_name}/nodes.feather",
                    f"layers/{layer_name}/edges.npz",
                ]
            )
        elif layer_info["type"] == "points":
            required_files.append(f"layers/{layer_name}/nodes.feather")

        missing_files.extend([f for f in required_files if f not in files])

    # Check annotations
    for anno_name in metadata["structure"]["annotations"]:
        required_files = [
            f"annotations/{anno_name}/meta.json",
            f"annotations/{anno_name}/nodes.feather",
        ]
        missing_files.extend([f for f in required_files if f not in files])

    # Check linkage references valid layers
    if "linkage" in metadata:
        all_layer_names = set(metadata["structure"]["layers"].keys()) | set(
            metadata["structure"]["annotations"].keys()
        )
        for linkage_pair in metadata["linkage"]:
            for layer_name in linkage_pair:
                if layer_name not in all_layer_names:
                    raise ValueError(f"Linkage references unknown layer: {layer_name}")

    if missing_files:
        raise ValueError(f"Missing required files: {missing_files}")


def _process_layers(metadata, files, tf, nrn) -> MeshWorkSync:
    """Process layer data from tar file."""
    for layer_name, layer_info in metadata["structure"]["layers"].items():
        match layer_info["type"]:
            case "skeleton":
                nrn = build_skeleton(
                    parse_skeleton_files(layer_name, files, tf), nrn=nrn
                )
            case "graph":
                nrn = build_graph(parse_graph_files(layer_name, files, tf), nrn=nrn)
            case "mesh":
                # mesh = build_mesh(parse_mesh_files(layer_name, files, tf))
                pass
            case "points":
                # pcd = build_point_cloud(
                # parse_point_cloud_files(layer_name, files, tf)
                # )
                pass
    return nrn


def _process_annotations(metadata, files, tf, nrn) -> MeshWorkSync:
    """Process annotation data from tar file."""
    for anno_name, anno_info in metadata["structure"]["annotations"].items():
        nrn = build_point_cloud(
            parse_point_cloud_files(anno_name, files, tf, as_annotation=True),
            nrn=nrn,
            as_annotation=True,
        )
    return nrn


def _process_linkage(metadata, tf, nrn) -> MeshWorkSync:
    """Process linkage data from tar file."""
    linkages = metadata["linkage"]
    for linkage_pair in linkages:
        nrn = build_linkage(linkage_pair, tf, nrn=nrn)
    return nrn


class MeshworkIO:
    def __init__(self, path, use_https: bool = True):
        self.path = path
        if parse.urlparse(path).scheme in ["s3", "gs", "https", "http"]:
            self.cf = cloudfiles.CloudFiles(path, use_https=use_https)
            self._remote = True
        elif parse.urlparse(path).scheme == "mem":
            self.cf = cloudfiles.CloudFiles(path)
            self._remote = False
        else:
            self.cf = cloudfiles.CloudFiles(
                "file://" + str((Path(path).expanduser().absolute()))
            )
            self._remote = False
        self._saveable = parse.urlparse(self.cf.cloudpath).scheme in PUTABLE_SCHEMES

    @property
    def saveable(self):
        return self._saveable

    @property
    def remote(self):
        return self._remote

    def save(
        self,
        nrn: MeshWorkSync,
        filename: Optional[str] = None,
        allow_overwrite: bool = False,
    ):
        if not self._saveable:
            raise ValueError(
                f"Path {self.path} is not writable. Please provide a path with one of the following schemes: {PUTABLE_SCHEMES}"
            )
        if filename is None:
            filename = f"{nrn.name}.mwk"
        if not allow_overwrite:
            # Split up to avoid network
            if self.cf.exists(filename):
                raise FileExistsError(
                    f"{filename} already exists in path {self.cf.cloudpath}."
                )
        tar_bytes = _export_neuron_to_bytes(nrn)
        self.cf.put(filename, tar_bytes)

    def load(self, filename: str) -> MeshWorkSync:
        f = self.cf.get(filename, raw=True)
        if not f:
            raise FileNotFoundError(
                f"{filename} not found in path {self.cf.cloudpath}."
            )
        return self._import_neuron(f)

    def _import_neuron(self, file: bytes) -> MeshWorkSync:
        return _import_neuron_from_bytes(file)


def load_dict(tinfo, tf) -> dict:
    return orjson.loads(tf.extractfile(tinfo).read())


def load_dataframe(tinfo, tf) -> pd.DataFrame:
    df_buf = pa.BufferReader(tf.extractfile(tinfo).read())
    return pd.read_feather(df_buf)


def load_array(tinfo, tf) -> np.ndarray:
    buf = io.BytesIO(tf.extractfile(tinfo).read())
    return np.load(buf)["data"]


def load_sparse_matrix(tinfo, tf) -> "scipy.sparse.csgraph.csr_matrix":
    buf = io.BytesIO(tf.extractfile(tinfo).read())
    return load_npz(buf)


def export_linkage(nrn, tf) -> None:
    datapath = f"linkage"
    unique_linkage = np.unique(
        [sorted(b) for b in list(nrn._morphsync._links.keys())], axis=0
    ).tolist()
    for linkage_pair in unique_linkage:
        add_file_to_tar(
            name=f"{datapath}/{linkage_pair[0]}/{linkage_pair[1]}/linkage.feather",
            data=bytesio_feather(nrn._morphsync._links[tuple(linkage_pair)]),
            tf=tf,
        )


def export_skeleton_layer(
    layer,
    tf,
) -> None:
    datapath = f"layers/{layer.name}"
    add_file_to_tar(
        name=f"{datapath}/meta.json",
        data=dict_to_bytesio_json(
            {
                "spatial_columns": layer.spatial_columns,
                "name": layer.name,
                "root": layer.root,
            }
        ),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/nodes.feather",
        data=bytesio_feather(layer.nodes),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/edges.npz",
        data=bytesio_array(layer.edges),
        tf=tf,
    )
    save_skeleton_base_properties(
        datapath,
        layer._base_properties,
        tf,
    )


def export_graph_layer(
    layer,
    tf,
) -> None:
    datapath = f"layers/{layer.name}"
    add_file_to_tar(
        name=f"{datapath}/meta.json",
        data=dict_to_bytesio_json(
            {"spatial_columns": layer.spatial_columns, "name": layer.name}
        ),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/nodes.feather",
        data=bytesio_feather(layer.nodes),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/edges.npz",
        data=bytesio_array(layer.edges),
        tf=tf,
    )


def export_point_cloud_layer(
    layer,
    tf,
    as_annotation: bool = True,
) -> None:
    if as_annotation:
        datapath = f"annotations/{layer.name}"
    else:
        datapath = f"layers/{layer.name}"
    add_file_to_tar(
        name=f"{datapath}/meta.json",
        data=dict_to_bytesio_json(
            {"spatial_columns": layer.spatial_columns, "name": layer.name}
        ),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/nodes.feather",
        data=bytesio_feather(layer.nodes),
        tf=tf,
    )


def export_mesh_layer(layer, tf) -> None:
    datapath = f"layers/{layer.name}"
    add_file_to_tar(
        name=f"{datapath}/meta.json",
        data=dict_to_bytesio_json(
            {"spatial_columns": layer.spatial_columns, "name": layer.name}
        ),
        tf=tf,
    )


def extract_metadata(nrn) -> bytes:
    layer_structure = {l.name: {"type": l.layer_type} for l in nrn.layers}
    annotation_structure = {a.name: {"type": a.layer_type} for a in nrn.annotations}
    all_links = [sorted(b) for b in list(nrn._morphsync._links.keys())]
    linkages = np.unique([sorted(b) for b in all_links], axis=0).tolist()
    metadata = {
        "name": nrn.name,
        "meta": nrn.meta,
        "file_version": 1.0,
        "structure": {"layers": layer_structure, "annotations": annotation_structure},
        "linkage": linkages,
    }
    metadata_bytes = orjson.dumps(
        metadata,
        option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
    )
    return metadata_bytes


def bytesio_feather(df, compression="zstd") -> bytes:
    buf = io.BytesIO()
    df.to_feather(buf, compression=compression)
    return buf.getvalue()


def bytesio_sparse_matrix(sparse_matrix) -> bytes:
    buf = io.BytesIO()
    save_npz(buf, sparse_matrix)
    return buf.getvalue()


def bytesio_array(data) -> bytes:
    buf = io.BytesIO()
    savez_compressed(buf, data=data)
    return buf.getvalue()


def dict_to_bytesio_json(d) -> bytes:
    return orjson.dumps(
        d,
        option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
    )


def add_file_to_tar(name, data, tf) -> bytes:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def save_skeleton_base_properties(
    path, base_properties, tf, matrix_properties=["base_csgraph", "base_csgraph_binary"]
) -> None:
    datapath = f"{path}/base_properties"

    add_file_to_tar(
        name=f"{datapath}/properties.json",
        data=dict_to_bytesio_json(
            {k: v for k, v in base_properties.items() if k not in matrix_properties}
        ),
        tf=tf,
    )
    for v in matrix_properties:
        add_file_to_tar(
            name=f"{datapath}/properties_{v}.npz",
            data=bytesio_sparse_matrix(base_properties[v]),
            tf=tf,
        )


def parse_skeleton_files(layer_name, files, tf) -> dict:
    prefix = f"layers/{layer_name}"
    skeleton_parts = {}
    for fn in files.keys():
        if fn.startswith(prefix):
            path_parts = Path(fn).parts
            match (path_parts[-1], path_parts[-2]):
                case ("meta.json", layer_name):
                    skeleton_parts["meta"] = load_dict(files[fn], tf)
                case ("nodes.feather", layer_name):
                    skeleton_parts["nodes"] = load_dataframe(files[fn], tf)
                case ("edges.npz", layer_name):
                    skeleton_parts["edges"] = load_array(files[fn], tf)
                case ("properties.json", "base_properties"):
                    skeleton_parts["base_properties"] = load_dict(files[fn], tf)
                case ("properties_base_csgraph.npz", "base_properties"):
                    skeleton_parts["base_csgraph"] = load_sparse_matrix(files[fn], tf)
                case ("properties_base_csgraph_binary.npz", "base_properties"):
                    skeleton_parts["base_csgraph_binary"] = load_sparse_matrix(
                        files[fn], tf
                    )
    return skeleton_parts


def build_skeleton(
    skeleton_parts: dict, nrn: "MeshWorkSync" = None
) -> Union[SkeletonSync, MeshWorkSync]:
    inherited_properties = {}
    inherited_properties.update(skeleton_parts["base_properties"])
    inherited_properties["base_csgraph"] = skeleton_parts.get("base_csgraph")
    inherited_properties["base_csgraph_binary"] = skeleton_parts.get(
        "base_csgraph_binary"
    )
    if nrn is None:
        skel_sync = SkeletonSync(
            name=skeleton_parts["meta"]["name"],
            vertices=skeleton_parts["nodes"],
            edges=skeleton_parts["edges"],
            root=skeleton_parts["meta"]["root"],
            spatial_columns=skeleton_parts["meta"]["spatial_columns"],
        )
        skel_sync._set_base_properties(inherited_properties)
        return skel_sync
    else:
        nrn.add_skeleton(
            vertices=skeleton_parts["nodes"],
            edges=skeleton_parts["edges"],
            root=skeleton_parts["meta"]["root"],
            spatial_columns=skeleton_parts["meta"]["spatial_columns"],
        )
        nrn.layers[skeleton_parts["meta"]["name"]]._set_base_properties(
            inherited_properties
        )
        return nrn


def parse_mesh_files(layer_name, files, tf):
    pass


def build_mesh():
    pass


def parse_graph_files(layer_name, files, tf) -> dict:
    prefix = f"layers/{layer_name}"
    graph_parts = {}
    for fn in files.keys():
        if fn.startswith(prefix):
            path_parts = Path(fn).parts
            match (path_parts[-1], path_parts[-2]):
                case ("meta.json", layer_name):
                    graph_parts["meta"] = load_dict(files[fn], tf)
                case ("nodes.feather", layer_name):
                    graph_parts["nodes"] = load_dataframe(files[fn], tf)
                case ("edges.npz", layer_name):
                    graph_parts["edges"] = load_array(files[fn], tf)
    return graph_parts


def build_graph(
    graph_parts: dict, nrn: MeshWorkSync = None
) -> Union[GraphSync, MeshWorkSync]:
    if nrn is None:
        graph_sync = GraphSync(
            name=graph_parts["meta"]["name"],
            vertices=graph_parts["nodes"],
            edges=graph_parts["edges"],
            spatial_columns=graph_parts["meta"]["spatial_columns"],
        )
        return graph_sync
    else:
        nrn.add_graph(
            vertices=graph_parts["nodes"],
            edges=graph_parts["edges"],
            spatial_columns=graph_parts["meta"]["spatial_columns"],
        )
        return nrn


def parse_point_cloud_files(layer_name, files, tf, as_annotation: bool = True) -> dict:
    if as_annotation:
        prefix = f"annotations/{layer_name}"
    else:
        prefix = f"layers/{layer_name}"
    pcd_parts = {}
    for fn in files.keys():
        if fn.startswith(prefix):
            path_parts = Path(fn).parts
            match (path_parts[-1], path_parts[-2]):
                case ("meta.json", layer_name):
                    pcd_parts["meta"] = load_dict(files[fn], tf)
                case ("nodes.feather", layer_name):
                    pcd_parts["nodes"] = load_dataframe(files[fn], tf)
    return pcd_parts


def build_point_cloud(
    pcd_parts, nrn: "MeshWorkSync" = None, as_annotation: bool = True
) -> Union[PointCloudSync, MeshWorkSync]:
    if nrn is None:
        point_cloud_sync = PointCloudSync(
            name=pcd_parts["meta"]["name"],
            vertices=pcd_parts["nodes"],
            spatial_columns=pcd_parts["meta"]["spatial_columns"],
        )
        return point_cloud_sync
    else:
        if as_annotation:
            nrn.add_point_annotations(
                name=pcd_parts["meta"]["name"],
                vertices=pcd_parts["nodes"],
                spatial_columns=pcd_parts["meta"]["spatial_columns"],
            )
        return nrn


def build_linkage(
    linkage_pair,
    tf,
    nrn,
) -> MeshWorkSync:
    prefix = f"linkage/{linkage_pair[0]}/{linkage_pair[1]}"
    link_df = load_dataframe(f"{prefix}/linkage.feather", tf)
    # Determine source based on the length of the vertices in the mapping and in the skeleton layer
    if len(link_df) == len(nrn._all_objects[linkage_pair[0]].nodes):
        source_layer = linkage_pair[0]
        target_layer = linkage_pair[1]
    elif len(link_df) == len(nrn._all_objects[linkage_pair[1]].nodes):
        source_layer = linkage_pair[1]
        target_layer = linkage_pair[0]
    else:
        raise ValueError("Linkage DataFrame does not match any layer.")

    layer = nrn._all_objects[source_layer]
    nrn._all_objects[source_layer]._process_linkage(
        Link(
            link_df.set_index(source_layer).loc[layer.vertex_index][target_layer],
            source=source_layer,
            target=target_layer,
        )
    )
    return nrn
