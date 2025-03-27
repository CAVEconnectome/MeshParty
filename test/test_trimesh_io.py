import itertools
import tempfile

import numpy
import pytest

from meshparty import trimesh_io
from .basic_test import build_basic_cube_mesh

io_file_exts = [".h5", ".obj"]
io_overwrite_flags = [True, False]
io_file_exist = [True, False]


def write_meshobject_h5(mesh, filename, overwrite):
    trimesh_io.write_mesh_h5(
        filename,
        mesh.vertices,
        mesh.faces,
        mesh.face_normals,
        link_edges=mesh.link_edges,
        overwrite=overwrite,
    )


def write_meshobject_other(mesh, filename, overwrite):
    mesh.write_to_file(filename)


@pytest.mark.parametrize(
    "file_ext,overwrite_flag,file_exist",
    itertools.product(io_file_exts, io_overwrite_flags, io_file_exist),
)
def test_file_read_write(basic_mesh, file_ext, overwrite_flag, file_exist):
    m = basic_mesh
    write_func = {".h5": write_meshobject_h5}.get(file_ext, write_meshobject_other)

    # leave file around
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=(not file_exist)) as tf:
        fname = tf.name

    if file_exist and not overwrite_flag and file_ext == ".h5":
        with pytest.raises(FileExistsError):
            write_func(m, fname, overwrite_flag)
    else:
        write_func(m, fname, overwrite_flag)

    # write func fails silently if writing h5 without overwrite.
    #   obj always overwrites
    if file_exist and not overwrite_flag and file_ext == ".h5":
        with pytest.raises(OSError):
            mvtx, mfaces, mnormals, link_edges, node_mask = trimesh_io.read_mesh(fname)
    else:
        mvtx, mfaces, mnormals, mlink_edges, node_mask = trimesh_io.read_mesh(fname)
        assert numpy.array_equal(mvtx, m.vertices)
        assert numpy.array_equal(mfaces, m.faces)

        # not sure why, but normals are not being returned in obj
        if file_ext != ".obj":
            assert numpy.array_equal(mnormals, m.face_normals)
            assert numpy.array_equal(mlink_edges, m.link_edges)


@pytest.mark.parametrize(
    "indicator_fstring,propstring",
    [
        ("meshparty.utils.create_csgraph", "csgraph"),
        ("meshparty.utils.create_nxgraph", "nxgraph"),
    ],
)
def test_lazy_mesh_props(basic_mesh, indicator_fstring, propstring, mocker):
    """
    indicator_fstring:
        function mocked indicating lazy property is being evaluated
    propstring:
        lazily evaluated property being tested against
    """
    m = basic_mesh

    mocked_f = mocker.patch(indicator_fstring)

    mocked_f.assert_not_called()

    firstp = getattr(m, propstring)
    mocked_f.assert_called_once()

    secondp = getattr(m, propstring)
    mocked_f.assert_called_once()

    assert firstp is secondp


@pytest.fixture(scope="function")
def basic_cube_mesh_fscope():
    with build_basic_cube_mesh() as r:
        yield r
