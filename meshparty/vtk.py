import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import numpy as np


def trimesh_to_vtk(vertices, tris):
    r"""Return a `vtkPolyData` representation of a :map:`TriMesh` instance
    Parameters
    ----------
    vertices : numpy array of Nx3 vertex positions (x,y,z)
    tris: numpy array of Mx3 triangle vertex indices (int64)
    Returns
    -------
    `vtk_mesh` : `vtkPolyData`
        A VTK mesh representation of the Menpo :map:`TriMesh` data
    Raises
    ------
    ValueError:
        If the input trimesh is not 3D.
    """

    if tris.shape[1] != 3:
        raise ValueError('trimesh_to_vtk() only works on 3D TriMesh instances')

    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=1))
    mesh.SetPoints(points)

    cells = vtk.vtkCellArray()

    # Seemingly, VTK may be compiled as 32 bit or 64 bit.
    # We need to make sure that we convert the trilist to the correct dtype
    # based on this. See numpy_to_vtkIdTypeArray() for details.
    isize = vtk.vtkIdTypeArray().GetDataTypeSize()
    req_dtype = np.int32 if isize == 4 else np.int64
    n_tris = tris.shape[0]
    cells.SetCells(n_tris,
                   numpy_to_vtkIdTypeArray(
                       np.hstack((np.ones(n_tris)[:, None] * 3,
                                  tris)).astype(req_dtype).ravel(),
                       deep=1))
    mesh.SetPolys(cells)
    return mesh


