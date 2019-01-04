import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import numpy as np


def numpy_to_vtk_cells(mat):
    cells = vtk.vtkCellArray()

    # Seemingly, VTK may be compiled as 32 bit or 64 bit.
    # We need to make sure that we convert the trilist to the correct dtype
    # based on this. See numpy_to_vtkIdTypeArray() for details.
    isize = vtk.vtkIdTypeArray().GetDataTypeSize()
    req_dtype = np.int32 if isize == 4 else np.int64
    n_elems = mat.shape[0]
    n_dim = mat.shape[1]
    cells.SetCells(n_elems,
                   numpy_to_vtkIdTypeArray(
                       np.hstack((np.ones(n_elems)[:, None] * n_dim,
                                  mat)).astype(req_dtype).ravel(),
                       deep=1))
    return cells


def numpy_rep_to_vtk(vertices, shapes, edges=None):
    """ converts a numpy representation of vertices and vertex connection graph
      to a polydata object and corresponding cell array

        :param vertices: a Nx3 numpy array of vertex locations
        :param shapes: a MxK numpy array of vertex connectivity
                       (could be triangles (K=3) or edges (K=2))

        :return: (vtkPolyData, vtkCellArray)
        a polydata object with point set according to vertices,
        and a vtkCellArray of the shapes
    """

    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=1))
    mesh.SetPoints(points)

    cells = numpy_to_vtk_cells(shapes)
    if edges is not None:
        edges = numpy_to_vtk_cells(edges)

    return mesh, cells, edges


def graph_to_vtk(vertices, edges):
    """ converts a numpy representation of vertices and edges
      to a vtkPolyData object

        :param vertices: a Nx3 numpy array of vertex locations
        :param eges: a Mx2 numpy array of vertex connectivity
        where the values are the indexes of connected vertices

        :return: vtkPolyData
        a polydata object with point set according to vertices
        and edges as its Lines

        :raises: ValueError
        if edges is not 2d or refers to out of bounds vertices
    """
    if edges.shape[1] != 2:
        raise ValueError('graph_to_vtk() only works on edge lists')
    if np.max(edges) >= len(vertices):
        msg = 'edges refer to non existent vertices {}.'
        raise ValueError(msg.format(np.max(edges)))
    mesh, cells, edges = numpy_rep_to_vtk(vertices, edges)
    mesh.SetLines(cells)
    return mesh


def trimesh_to_vtk(vertices, tris, mesh_edges=None):
    """Return a `vtkPolyData` representation of a :map:`TriMesh` instance
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
        If the input trimesh is not 3D
        or tris refers to out of bounds vertex indices
    """

    if tris.shape[1] != 3:
        raise ValueError('trimesh_to_vtk() only works on 3D TriMesh instances')
    if np.max(tris) >= len(vertices):
        msg = 'edges refer to non existent vertices {}.'
        raise ValueError(msg.format(np.max(tris)))
    mesh, cells, edges = numpy_rep_to_vtk(vertices, tris, mesh_edges)
    mesh.SetPolys(cells)
    if edges is not None:
        mesh.SetLines(edges)

    return mesh


def vtk_cellarray_to_shape(vtk_cellarray, ncells):
    cellarray = vtk_to_numpy(vtk_cellarray)
    cellarray = cellarray.reshape(ncells, int(len(cellarray)/ncells))
    return cellarray[:, 1:]


def decimate_trimesh(trimesh, reduction=.1):
    poly = trimesh_to_vtk(trimesh.vertices, trimesh.faces)
    dec = vtk.vtkDecimatePro()
    dec.SetTargetReduction(reduction)
    dec.PreserveTopologyOn()
    dec.SetInputData(poly)
    dec.Update()
    out_poly = dec.GetOutput()

    points = vtk_to_numpy(out_poly.GetPoints().GetData())
    ntris = out_poly.GetNumberOfPolys()
    tris = vtk_cellarray_to_shape(out_poly.GetPolys().GetData(), ntris)
    return points, tris


def remove_unused_verts(mesh):
    used_verts=np.unique(mesh.faces.ravel())
    new_verts=mesh.vertices[used_verts,:]
    new_face = np.zeros(mesh.faces.shape)
    for i in range(3):   
        new_face[:,i]=np.searchsorted(used_verts, mesh.faces[:,i])
    return new_verts, new_face


def filter_largest_cc(trimesh):
    poly = trimesh_to_vtk(trimesh.vertices, trimesh.faces, trimesh.mesh_edges)
    connf = vtk.vtkConnectivityFilter()
    connf.SetInputData(poly)
    connf.SetExtractionModeToLargestRegion()
    connf.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(connf.GetOutputPort())
    clean.PointMergingOff()
    clean.Update()

    out_poly = clean.GetOutput()
    points = vtk_to_numpy(out_poly.GetPoints().GetData())
    ntris = out_poly.GetNumberOfPolys()
    tris = vtk_cellarray_to_shape(out_poly.GetPolys().GetData(), ntris)
    nedges = out_poly.GetNumberOfLines()
    if nedges > 0:
        edges = vtk_cellarray_to_shape(out_poly.GetLines().GetData(), nedges)
    else:
        edges = None
    return points, tris, edges


def calculate_cross_sections(mesh, graph_verts, graph_edges, calc_centers=True):

    mesh_polydata = trimesh_to_vtk(mesh.vertices, mesh.faces)

    cutter = vtk.vtkPlaneCutter()
    cutter.SetInputData(mesh_polydata)
    plane = vtk.vtkPlane()
    cd = vtk.vtkCleanPolyData()
    cf = vtk.vtkPolyDataConnectivityFilter()
    cf.SetInputConnection(cd.GetOutputPort())
    cf.SetExtractionModeToClosestPointRegion()
    cutter.SetPlane(plane)
    cutStrips = vtk.vtkStripper()
    cutStrips.JoinContiguousSegmentsOn()
    cutStrips.SetInputConnection(cf.GetOutputPort())

    cross_sections = np.zeros(len(graph_edges), dtype=np.float)

    if calc_centers:
        centers = np.zeros((len(graph_edges), 3))

    massfilter = vtk.vtkMassProperties()
    massfilter.SetInputConnection(cutter.GetOutputPort())
    t = vtk.vtkTriangleFilter()
    dvs = graph_verts[graph_edges[:, 0], :]-graph_verts[graph_edges[:, 1], :]
    dvs = (dvs / np.linalg.norm(dvs, axis=1)[:, np.newaxis])
    for k, edge in enumerate(graph_edges):
        dv = dvs[k, :]

        dv = dv.tolist()

        v = graph_verts[graph_edges[k, 0], :]
        v = v.tolist()
        plane.SetNormal(*dv)
        plane.SetOrigin(*v)

        cutter.Update()
        pd = cutter.GetOutputDataObject(0).GetBlock(0).GetPiece(0)

        cd.SetInputData(pd)
        cf.SetClosestPoint(*v)
        cutStrips.Update()

        cutPoly = vtk.vtkPolyData()
        cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
        cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

        t.SetInputData(cutPoly)
        if calc_centers:
            pts = vtk_to_numpy(cf.GetOutput().GetPoints().GetData())
            # centerOfMassFilter = vtk.vtkCenterOfMass()
            # centerOfMassFilter.SetInputConnection(t.GetOutputPort())
            # centerOfMassFilter.Update()
            centers[k,:]=np.mean(pts, axis=0)

        massfilter = vtk.vtkMassProperties()
        massfilter.SetInputConnection(t.GetOutputPort())
        massfilter.Update()

        cross_sections[k] = massfilter.GetSurfaceArea()

    return cross_sections, centers

def make_vtk_skeleton_from_paths(verts, paths):
    cell_list =[]
    num_cells=0
    for p in paths: 
        cell_list.append(len(p))
        cell_list+=p
        num_cells+=1
    cell_array = np.array(cell_list)

    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts, deep=1))
    mesh.SetPoints(points)

    cells = vtk.vtkCellArray()

    # Seemingly, VTK may be compiled as 32 bit or 64 bit.
    # We need to make sure that we convert the trilist to the correct dtype
    # based on this. See numpy_to_vtkIdTypeArray() for details.
    isize = vtk.vtkIdTypeArray().GetDataTypeSize()
    req_dtype = np.int32 if isize == 4 else np.int64

    cells.SetCells(num_cells,
                   numpy_to_vtkIdTypeArray(cell_array,deep=1))
    mesh.SetLines(cells)
    return mesh