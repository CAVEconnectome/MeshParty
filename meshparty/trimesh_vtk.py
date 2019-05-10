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


def trimesh_to_vtk(vertices, tris, graph_edges=None):
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
    mesh, cells, edges = numpy_rep_to_vtk(vertices, tris, graph_edges)
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
    poly = trimesh_to_vtk(trimesh.vertices, trimesh.faces, trimesh.graph_edges)
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


def vtk_super_basic(actors, camera=None, do_save=False, filename=None, scale=4, back_color=(.1, .1, .1),
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
    if do_save:
        assert(filepath is not None)
    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    if camera is not None:
        ren.SetActiveCamera(camera)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(VIDEO_WIDTH, VIDEO_HEIGHT)
    # renderWindow.SetAlphaBitPlanes(1)

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
    else:
        ren.SetActiveCamera(camera)
        ren.ResetCameraClippingRange()
        camera.ViewingRaysModified()
    renWin.Render()


    if do_save is False:
        trackCamera = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(trackCamera)
        # enable user interface interactor
        iren.Initialize()
        iren.Render()
        iren.Start()


    if do_save is True:
        renWin.OffScreenRenderingOn()
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetScale(scale)
        w2if.SetInput(renWin)
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()

    renWin.Finalize()

    return ren


def vtk_camera_from_quat(pos_nm, orient_quat, camera_distance=10000, ngl_correct=True):
    """define a vtk camera with a particular orientation
        Parameters
        ----------
        pos_nm: np.array, list, tuple
            an iterator of length 3 containing the focus point of the camera
        orient_quat: np.array, list, tuple
            a len(4) quatenerion (x,y,z,w) describing the rotation of the camera
            such as returned by neuroglancer x,y,z,w all in [0,1] range
        camera_distance: float
            the desired distance from pos_nm to the camera (default = 10000 nm)

        Returns
        -------
        vtk.vtkCamera
            a vtk camera setup according to these rules
    """
    camera = vtk.vtkCamera()
    # define the quaternion in vtk, note the swapped order
    # w,x,y,z instead of x,y,z,w
    quat_vtk=vtk.vtkQuaterniond(orient_quat[3],
                                orient_quat[0],
                                orient_quat[1],
                                orient_quat[2])
    # use this to define a rotation matrix in x,y,z
    # right handed units
    M = np.zeros((3, 3), dtype=np.float32)
    quat_vtk.ToMatrix3x3(M)
    # the default camera orientation is y up
    up = [0, 1, 0]
    # calculate default camera position is backed off in positive z 
    pos = [0, 0, camera_distance]
    
    # set the camera rototation by applying the rotation matrix
    camera.SetViewUp(*np.dot(M,up))
    # set the camera position by applying the rotation matrix
    camera.SetPosition(*np.dot(M,pos))
    if ngl_correct:
        # neuroglancer has positive y going down
        # so apply these azimuth and roll corrections
        # to fix orientatins
        camera.Azimuth(-180)
        camera.Roll(180)

    # shift the camera posiiton and focal position
    # to be centered on the desired location
    p=camera.GetPosition()
    p_new = np.array(p)+pos_nm
    camera.SetPosition(*p_new)
    camera.SetFocalPoint(*pos_nm)
    return camera

def vtk_camera_from_ngl_state(state_d, zoom_factor=300.0):
    """define a vtk camera from a neuroglancer state dictionary
        Parameters
        ----------
        state_d: dict
            an neuroglancer state dictionary
        zoom_factor: float
            how much to multiply zoom by to get camera backoff distance
            default = 300 > ngl_zoom = 1 > 300 nm backoff distance

        Returns
        -------
        vtk.vtkCamera
            a vtk camera setup that mathces this state
    """

    orient = state_d.get('perspectiveOrientation', [0.0,0.0,0.0,1.0])
    zoom = state_d.get('perspectiveZoom', 10.0)
    position = state_d['navigation']['pose']['position']
    pos_nm = np.array(position['voxelCoordinates'])*position['voxelSize']
    camera = vtk_camera_from_quat(pos_nm, orient, zoom*zoom_factor, ngl_correct=True)
    
    return camera


def make_mesh_actor(mesh, color=(0, 1, 0),
                    opacity=0.1,
                    vertex_scalars=None,
                    lut=None,
                    calc_normals=True):

    mesh_poly = trimesh_to_vtk(mesh.vertices, mesh.faces, mesh.graph_edges)
    if vertex_scalars is not None:
        mesh_poly.GetPointData().SetScalars(numpy_to_vtk(vertex_scalars))
    mesh_mapper = vtk.vtkPolyDataMapper()
    if calc_normals and mesh.graph_edges is None:
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
                       vertex_property=None,
                       vertex_data=None,
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

    data = None
    if vertex_data is None and vertex_property is not None:
        data = sk.vertex_properties[vertex_property]
    else:
        data = vertex_data

    if data is not None:
        if normalize_property:
            data = data / np.nanmax(data)
        sk_mesh.GetPointData().SetScalars(numpy_to_vtk(data))
        lut = vtk.vtkLookupTable()
        if lut_map is not None:
            lut_map(lut)
        lut.Build()
        mapper.ScalarVisibilityOn()
        mapper.SetLookupTable(lut)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(line_width)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)
    return actor


def neuron_actors(mesh, pre_syn_positions=None, post_syn_positions=None,
                  mesh_color=(0.459, 0.439, 0.702), pre_color=(0.994, 0.098, 0.106), post_color=(0.176, 0.996, 0.906),
                  mesh_opacity=0.8, pre_opacity=1, post_opacity=1,
                  pre_size=400, post_size=400):
    mesh_actor = make_mesh_actor(mesh, color=mesh_color, opacity=mesh_opacity)
    nrn_act = [mesh_actor]
    if pre_syn_positions is not None:
        pre_actor = make_point_cloud_actor(pre_syn_positions, size=pre_size, color=pre_color, opacity=pre_opacity)
        nrn_act.append(pre_actor)
    if post_syn_positions is not None:
        post_actor = make_point_cloud_actor(post_syn_positions, size=post_size, color=post_color, opacity=post_opacity)
        nrn_act.append(post_actor)
    return nrn_act


def make_point_cloud_actor(xyz,
                           size=100,
                           color=(0,0,0),
                           opacity=0.5):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))

    scales = vtk.vtkFloatArray()
    scales.SetName('scale')

    clr = vtk.vtkFloatArray()
    clr.SetName('color')

    colormap = vtk.vtkLookupTable()
    if np.array(color).shape == (3,):
        single_color = True
        colormap.SetNumberOfTableValues(1)
        colormap.SetTableValue(0, color[0], color[1], color[2], opacity)
    else:
        single_color = False
        colormap.SetNumberOfTableValues(len(color))
        for ii, color_row in enumerate(color):
            colormap.SetTableValue(ii, color_row[0], color_row[1], color_row[2], opacity)
        
    if np.isscalar(size):
        size = np.full(len(xyz), size)
    elif len(size) != len(xyz):
        raise ValueError('Size must be either a scalar or an len(xyz) x 1 array')
    for ii in range(len(xyz)):
        scales.InsertNextValue(size[ii])
        if single_color:
            clr.InsertNextValue(0)
        else:
            clr.InsertNextValue(ii)
            
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.GetPointData().AddArray(scales)
    grid.GetPointData().SetActiveScalars('scale')
    grid.GetPointData().AddArray(clr)

    ss = vtk.vtkSphereSource()
    ss.SetRadius(1)

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(grid)
    glyph.SetSourceConnection(ss.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    if single_color:
        mapper.SetScalarRange(0,0)
    else:
        mapper.SetScalarRange(1,len(xyz))
    mapper.SelectColorArray('color')
    mapper.SetLookupTable(colormap)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def vtk_linked_point_actor(vertices_a, inds_a,
                           vertices_b, inds_b,
                           line_width=1, color=(0, 0, 0), opacity=0.2):
    if len(inds_a) != len(inds_b):
        raise ValueError('Linked points must have the same length')

    link_verts = np.vstack((vertices_a[inds_a], vertices_b[inds_b]))
    link_edges = np.vstack((np.arange(len(inds_a)),
                            len(inds_a)+np.arange(len(inds_b))))
    link_poly = graph_to_vtk(link_verts, link_edges.T)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(link_poly)

    link_actor = vtk.vtkActor()
    link_actor.SetMapper(mapper)
    link_actor.GetProperty().SetLineWidth(line_width)
    link_actor.GetProperty().SetColor(color)
    link_actor.GetProperty().SetOpacity(opacity)
    return link_actor


def vtk_oriented_camera(center, up_vector=(0, -1, 0), backoff=500, backoff_vector=(0,0,1)):
    '''
    Generate a camera pointed at a specific location, oriented with a given up
    direction, set to a backoff.
    '''
    camera = vtk.vtkCamera()

    pt_center = center

    vup=np.array(up_vector)
    vup=vup/np.linalg.norm(vup)

    bv = np.array(backoff_vector)
    pt_backoff = pt_center - backoff * 1000 * bv

    camera.SetFocalPoint(*pt_center)
    camera.SetViewUp(*vup)
    camera.SetPosition(*pt_backoff)
    return camera
