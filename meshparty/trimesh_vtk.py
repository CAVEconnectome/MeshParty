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
        :param edges: a Mx2 numpy array of vertex connectivity
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


def remove_unused_verts(verts, faces):
    """removes unused vertices from a graph or mesh

    verts = NxD numpy array of vertex locations
    faces = MxK numpy array of connected shapes (i.e. edges or tris)
    (entries are indices into verts)

    returns: 
    new_verts, new_face
    a filtered set of vertices and reindexed set of faces
    """
    used_verts = np.unique(faces.ravel())
    new_verts = verts[used_verts, :]
    new_face = np.zeros(faces.shape, dtype=faces.dtype)
    for i in range(faces.shape[1]):
        new_face[:, i] = np.searchsorted(used_verts, faces[:, i])
    return new_verts, new_face


def vtk_poly_to_mesh_components(poly):
    points = vtk_to_numpy(poly.GetPoints().GetData())
    ntris = poly.GetNumberOfPolys()
    tris = vtk_cellarray_to_shape(poly.GetPolys().GetData(), ntris)
    nedges = poly.GetNumberOfLines()
    if nedges > 0:
        edges = vtk_cellarray_to_shape(poly.GetLines().GetData(), nedges)
    else:
        edges = None
    return points, tris, edges


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
    return vtk_poly_to_mesh_components(clean.GetOutput())


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
            centers[k, :] = np.mean(pts, axis=0)

        massfilter = vtk.vtkMassProperties()
        massfilter.SetInputConnection(t.GetOutputPort())
        massfilter.Update()

        cross_sections[k] = massfilter.GetSurfaceArea()

    return cross_sections, centers


def make_vtk_skeleton_from_paths(verts, paths):
    cell_list = []
    num_cells = 0
    for p in paths:
        cell_list.append(len(p))
        cell_list += p
        num_cells += 1
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
                   numpy_to_vtkIdTypeArray(cell_array, deep=1))
    mesh.SetLines(cells)
    return mesh


def vtk_super_basic(actors, camera=None, do_save=False, folder=".", back_color=(.1, .1, .1),
                    VIDEO_WIDTH=1080, VIDEO_HEIGHT=720):
    """
    Create a window, renderer, interactor, add the actors and start the thing

    Parameters
    ----------
    actors :  list of vtkActors

    Returns
    -------
    nothing
    """

    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(VIDEO_WIDTH, VIDEO_HEIGHT)
    if camera is not None:
        ren.SetActiveCamera(camera)

    ren.SetBackground(*back_color)
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)    

    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a)

    # render
    if camera is None:
        ren.ResetCamera()
    renWin.Render()

    trackCamera = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(trackCamera)
    # enable user interface interactor
    iren.Initialize()
    iren.Start()
    renWin.Finalize()

    return ren


def make_mesh_actor(mesh, color=(0, 1, 0),
                    opacity=0.1,
                    vertex_scalars=None,
                    lut=None,
                    calc_normals=True):

    mesh_poly = trimesh_to_vtk(mesh.vertices, mesh.faces, mesh.mesh_edges)
    if vertex_scalars is not None:
        mesh_poly.GetPointData().SetScalars(numpy_to_vtk(vertex_scalars))
    mesh_mapper = vtk.vtkPolyDataMapper()
    if calc_normals and mesh.mesh_edges is None:
        norms = vtk.vtkTriangleMeshPointNormals()
        norms.SetInputData(mesh_poly)
        mesh_mapper.SetInputConnection(norms.GetOutputPort())
    else:
        mesh_mapper.SetInputData(mesh_poly)
    mesh_actor = vtk.vtkActor()

    if lut is not None:
        mesh_mapper.SetLookupTable(lut)
    mesh_mapper.ScalarVisibilityOn()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().SetColor(*color)
    mesh_actor.GetProperty().SetOpacity(opacity)
    return mesh_actor


def vtk_skeleton_actor(sk,
                       edge_property=None,
                       normalize_property=True,
                       color=(0, 0, 0),
                       line_width=3,
                       opacity=0.7,
                       lut_map=None):
    sk_mesh = graph_to_vtk(sk.vertices, sk.edges)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(sk_mesh)
    if edge_property is not None:
        data = sk.edge_properties[edge_property]
        if normalize_property:
            data = data / np.nanmax(data)
        sk_mesh.GetCellData().SetScalars(numpy_to_vtk(data))
        lut = vtk.vtkLookupTable()
        if lut_map is not None:
            lut_map(lut)
        lut.Build()
        mapper.SetLookupTable(lut)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(line_width)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)
    return actor


def make_point_cloud_actor(xyz,
                           size=100,
                           color=(0, 0, 0),
                           opacity=0.5):

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))

    pc = vtk.vtkPolyData()
    pc.SetPoints(points)

    if np.isscalar(size):
        size = np.full(len(xyz), size)
    elif len(size) != len(xyz):
        raise ValueError('Size must be either a scalar or an len(xyz) x 1 array')
    pc.GetPointData().SetScalars(numpy_to_vtk(size))

    ss = vtk.vtkSphereSource()
    ss.SetRadius(1)

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(pc)

    glyph.SetSourceConnection(ss.GetOutputPort())
    glyph.SetScaleModeToScaleByScalar()
    glyph.ScalingOn()
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())

    actor = vtk.vtkActor()
    mapper.ScalarVisibilityOn()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)

    return actor

def vtk_linked_point_actor(vertices_a, inds_a,
                           vertices_b, inds_b,
                           line_width=1, color=(0, 0, 0), opacity=0.2):
    if len(inds_a) != len(inds_b):
        raise ValueError('Linked points must have the same length')

    link_verts = np.vstack((vertices_a[inds_a], vertices_b[inds_b]))
    link_edges = np.vstack((np.arange(len(inds_a)),
                            len(inds_a)+np.arange(len(inds_b))))
    link_poly = trimesh_vtk.graph_to_vtk(link_verts, link_edges.T)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(link_poly)

    link_actor = vtk.vtkActor()
    link_actor.SetMapper(mapper)
    link_actor.GetProperty().SetLineWidth(line_width)
    link_actor.GetProperty().SetColor(color)
    link_actor.GetProperty().SetOpacity(opacity)
    return link_actor
