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
        if len(edges)>0:
            edges = numpy_to_vtk_cells(edges)
        else:
            edges = None

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


def poly_to_mesh_components(poly):
    points = vtk_to_numpy(poly.GetPoints().GetData())
    ntris = poly.GetNumberOfPolys()
    if ntris > 0:
       tris = vtk_cellarray_to_shape(poly.GetPolys().GetData(), ntris)
    else:
        tris = None     
    nedges = poly.GetNumberOfLines()
    if nedges > 0:
        edges = vtk_cellarray_to_shape(poly.GetLines().GetData(), nedges)
    else:
        edges = None
    return points, tris, edges


def render_actors(actors, camera=None, do_save=False, filename=None,
                  scale=4, back_color=(1, 1, 1),
                  VIDEO_WIDTH=1080, VIDEO_HEIGHT=720):
    """
    Create a window, renderer, interactor, add the actors and start the thing

    Parameters
    ----------
    actors :  list[vtkActor]
        list of actors to render
    camera : vtkCamera
        camera to use for scence (optional..default to fit scene)
    do_save: bool
        write png image to disk, if false will open interactive window
    filename: str
        filepath to save png image to
    scale: 
        scale factor to use when saving images to disk (default 4) for higher res images
    back_color: Iterable
        rgb values (0,1) to determine for background color

    Returns
    -------
    vtk.vtkRenderer
        renderer when code was finished (useful for retrieving user input camera position ren.GetActiveCamera())
    """
    if do_save:
        assert(filename is not None)
    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren.UseFXAAOn()
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


def camera_from_quat(pos_nm, orient_quat, camera_distance=10000, ngl_correct=True):
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

def camera_from_ngl_state(state_d, zoom_factor=300.0):
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
    camera = camera_from_quat(pos_nm, orient, zoom*zoom_factor, ngl_correct=True)
    
    return camera


def process_colors(color,xyz):
    map_colors = False
    if not isinstance(color, np.ndarray):
        color = np.array(color)
    if color.shape == (len(xyz),3):
        # then we have explicit colors
        if color.dtype != np.uint8:
            # if not passing uint8 assume 0-1 mapping
            assert(np.max(color)<=1.0)
            assert(np.min(color)>=0)
            color = np.uint8(color*255)
    elif color.shape ==(len(xyz),):
        # then we want to map colors
        map_colors = True     
    elif color.shape == (3,):
        # then we have one explicit color
        assert(np.max(color)<=1.0)
        assert(np.min(color)>=0)
        car = np.array(color, dtype=np.uint8)*255 
        color = np.repeat(car[np.newaxis,:],len(xyz),axis=0)
    else:
        raise ValueError('color must have shapse Nx3 if explicitly setting, or (N,) if mapping, or (3,)')
    return color, map_colors

def mesh_actor(mesh,
               color=(0, 1, 0),
               opacity=0.1,
               vertex_colors=None,
               face_colors=None,
               lut=None,
               calc_normals=True,
               show_link_edges=False,
               line_width=3):

    if show_link_edges:
        mesh_poly = trimesh_to_vtk(mesh.vertices, mesh.faces, mesh.link_edges)
    else:
        mesh_poly = trimesh_to_vtk(mesh.vertices, mesh.faces, None)
    if vertex_colors is not None:
        vertex_color, map_vertex_color =  process_colors(vertex_colors, mesh.vertices)
        vtk_vert_colors = numpy_to_vtk(vertex_color)
        vtk_vert_colors.SetName('colors')
        mesh_poly.GetPointData().SetScalars(vtk_vert_colors)
    
    if face_colors is not None:
        face_color, map_face_colors = process_colors(face_colors, mesh.faces)
        vtk_face_colors = numpy_to_vtk(face_color)
        vtk_face_colors.SetName('colors')
        mesh_poly.GetCellData().SetScalars(vtk_face_colors)

    mesh_mapper = vtk.vtkPolyDataMapper()
    if calc_normals and (not show_link_edges):
        norms = vtk.vtkTriangleMeshPointNormals()
        norms.SetInputData(mesh_poly)
        mesh_mapper.SetInputConnection(norms.GetOutputPort())
    else:
        mesh_mapper.SetInputData(mesh_poly)

    mesh_actor = vtk.vtkActor()

    if lut is not None:
        mesh_mapper.SetLookupTable(lut)
        if face_colors is not None:
            if map_face_colors:
                mesh_mapper.SelectColorArray('colors')
    mesh_mapper.ScalarVisibilityOn()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().SetLineWidth(line_width)
    mesh_actor.GetProperty().SetColor(*color)
    mesh_actor.GetProperty().SetOpacity(opacity)
    return mesh_actor


def skeleton_actor(sk,
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

def point_cloud_actor(xyz,
                     size=100,
                     color=(0,0,0),
                     opacity=0.5):

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))

    pc = vtk.vtkPolyData() 
    pc.SetPoints(points)

    color, map_colors = process_colors(color, xyz)

    vtk_colors = numpy_to_vtk(color)
    vtk_colors.SetName('colors')

    if np.isscalar(size):
        size = np.full(len(xyz), size)
    elif len(size) != len(xyz):
        raise ValueError('Size must be either a scalar or an len(xyz) x 1 array')
    pc.GetPointData().SetScalars(numpy_to_vtk(size))
    pc.GetPointData().AddArray(vtk_colors)

    ss = vtk.vtkSphereSource()
    ss.SetRadius(1)

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(pc)
    glyph.SetInputArrayToProcess(3, 0, 0, 0, "colors")
    glyph.SetColorModeToColorByScalar()
    glyph.SetSourceConnection(ss.GetOutputPort())
    glyph.SetScaleModeToScaleByScalar()
    glyph.ScalingOn()
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    if map_colors:
        mapper.SetScalarRange(np.min(color), np.max(color))
        mapper.SelectColorArray('colors')
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def linked_point_actor(vertices_a, vertices_b,
                       inds_a=None, inds_b=None,
                       line_width=1, color=(0, 0, 0), opacity=0.2):
    if inds_a is None:
        inds_a = np.arange(len(vertices_a))
    if inds_b is None:
        inds_b = np.arange(len(vertices_b))

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


def oriented_camera(center, up_vector=(0, -1, 0), backoff=500, backoff_vector=(0,0,1)):
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
