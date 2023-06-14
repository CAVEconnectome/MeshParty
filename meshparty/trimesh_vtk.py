import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import numpy as np
import os
import logging
from meshparty.utils import remove_unused_verts

def numpy_to_vtk_cells(mat):
    """function to convert a numpy array of integers to a vtkCellArray

    Parameters
    ----------
    mat : np.array
        MxN array to be converted

    Returns
    -------
    vtk.vtkCellArray
        representing the numpy array, has the same shaped cell (N) at each of the M indices

    """

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

    Parameters
    ----------
    vertices: a Nx3 numpy array of vertex locations
    shapes: a MxK numpy array of vertex connectivity
                       (could be triangles (K=3) or edges (K=2))

    Returns
    -------
    vtk.vtkPolyData
        a polydata object with point set according to vertices,
    vtkCellArray
        a vtkCellArray of the shapes

    """

    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=1))
    mesh.SetPoints(points)

    cells = numpy_to_vtk_cells(shapes)
    if edges is not None:
        if len(edges) > 0:
            edges = numpy_to_vtk_cells(edges)
        else:
            edges = None

    return mesh, cells, edges


def graph_to_vtk(vertices, edges):
    """ converts a numpy representation of vertices and edges
      to a vtkPolyData object

    Parameters
    ----------
    vertices: np.array
        a Nx3 numpy array of vertex locations
    edges: np.array
        a Mx2 numpy array of vertex connectivity
        where the values are the indexes of connected vertices

    Returns
    -------
    vtk.vtkPolyData
        a polydata object with point set according to vertices
        and edges as its Lines

    Raises
    ------
    ValueError
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
    """Return a `vtkPolyData` representation of a :obj:`TriMesh` instance

    Parameters
    ----------
    vertices : np.array
        numpy array of Nx3 vertex positions (x,y,z)
    tris: np.array
        numpy array of Mx3 triangle vertex indices (int64)
    graph_edges: np.array
        numpy array of Kx2 of edges to set as the vtkPolyData.Lines

    Returns
    -------
    vtk_mesh : vtk.vtkPolyData
        A VTK mesh representation of the mesh :obj:`trimesh.TriMesh` data

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
    """Turn a vtkCellArray into a numpyarray of a fixed shape
    assumes your cell array has uniformed sized cells

    Parameters
    ----------
    vtk_cellarray : vtk.vtkCellArray
        a cell array to convert
    ncells: int
        how many cells are in array

    Returns
    -------
    np.array
        cellarray, a ncells x K array of cells, where K is the
        uniform shape of the cells.  Will error if cells are not uniform

    """
    cellarray = vtk_to_numpy(vtk_cellarray)
    cellarray = cellarray.reshape(ncells, int(len(cellarray)/ncells))
    return cellarray[:, 1:]


def decimate_trimesh(trimesh, reduction=.1):
    """ routine to decimate a mesh through vtk

    Parameters
    ----------
    trimesh : trimesh_io.Mesh
        a mesh to decimate
    reduction: float
        factor to decimate (default .1)

    Returns
    -------
    np.array
        points, the Nx3 mesh of vertices
    np.array
        tris, the Kx3 indices of faces

    """

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



def poly_to_mesh_components(poly):
    """ converts a vtkPolyData to its numpy components

    Parameters
    ----------
    poly : vtk.vtkPolyData
        a polydate object to convert to numpy components

    Returns
    -------
    np.array
        points, the Nx3 set of vertex locations
    np.array
        tris, the KxD set of faces (assumes a uniform cellarray)
    np.array
        edges, if exists uses the GetLines to make edges

    """
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
                  VIDEO_WIDTH=None, VIDEO_HEIGHT=None,
                  video_width=1080, video_height=720,
                  return_keyframes=False):
    """
    Visualize a set of actors in a 3d scene, optionally saving a snapshot. 
    Creates a window, renderer, interactor, add the actors and starts the visualization
    (can save images and close render window)

    Parameters
    ----------
    actors :  list[vtkActor]
        list of actors to render (see mesh_actor, point_cloud_actor, skeleton_actor)
    camera : :obj:`vtkCamera`
        camera to use for scence (optional..default to fit scene)
    do_save: bool
        write png image to disk, if false will open interactive window (default False)
    filename: str
        filepath to save png image to (default None)
    scale: 
        scale factor to use when saving images to disk (default 4) for higher res images
    back_color: Iterable
        rgb values (0,1) to determine for background color (default 1,1,1 = white)
    return_keyframes : bool
        whether to save a new camera as a keyframes when you press 'k' with window open

    Returns
    -------
    :obj:`vtk.vtkRenderer`
        renderer when code was finished
        (useful for retrieving user input camera position ren.GetActiveCamera())
    (list[vtk.vtkCamera])
        list of vtk cameras when user pressed 'k' (only if return_keyframes=True)

    """
    if do_save:
        assert(filename is not None)
    if VIDEO_HEIGHT is not None:
        logging.warning('VIDEO_HEIGHT deprecated, please use video_height')
        video_height=VIDEO_HEIGHT
    if VIDEO_WIDTH is not None:
        logging.warning('VIDEO_WIDTH is deprecated, please use VIDEO_WIDTH')
        video_width=VIDEO_WIDTH
    # create a rendering window and renderer
    ren, renWin, iren = _setup_renderer(
        video_width, video_height, back_color, camera=camera)
    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a)
    # render
    if camera is None:
        ren.ResetCamera()
    else:
        ren.ResetCameraClippingRange()
        camera.ViewingRaysModified()
    if return_keyframes:
        key_frame_cameras = []

        def vtkKeyPress(obj, event):
            key = obj.GetKeySym()
            if key == 'k':
                key_camera = vtk.vtkCamera()
                key_camera.DeepCopy(ren.GetActiveCamera())
                key_frame_cameras.append(key_camera)
            return
        iren.AddObserver("KeyPressEvent", vtkKeyPress)
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

    if return_keyframes:
        return ren, key_frame_cameras
    else:
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
    quat_vtk = vtk.vtkQuaterniond(orient_quat[3],
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
    camera.SetViewUp(*np.dot(M, up))
    # set the camera position by applying the rotation matrix
    camera.SetPosition(*np.dot(M, pos))
    if ngl_correct:
        # neuroglancer has positive y going down
        # so apply these azimuth and roll corrections
        # to fix orientatins
        camera.Azimuth(-180)
        camera.Roll(180)

    # shift the camera posiiton and focal position
    # to be centered on the desired location
    p = camera.GetPosition()
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

    orient = state_d.get('perspectiveOrientation', [0.0, 0.0, 0.0, 1.0])
    zoom = state_d.get('perspectiveZoom', 10.0)
    position = state_d['navigation']['pose']['position']
    pos_nm = np.array(position['voxelCoordinates'])*position['voxelSize']
    camera = camera_from_quat(pos_nm, orient, zoom *
                              zoom_factor, ngl_correct=True)

    return camera


def process_colors(color, xyz):
    """ utility function to normalize colors on an set of things
    Parameters
    ----------
    color : np.array
        a Nx3, or a N long, or a 3 long iterator the represents the 
        color or colors  you want to label xyz with
    xyz: np.array
        a NxD matrix you wish to 'color'
    Returns
    -------
    np.array
        a Nx3 or N long array of color values
    bool
        map_colors, whether the colors should be mapped through a colormap
        or used as is
    """
    map_colors = False
    if not isinstance(color, np.ndarray):
        color = np.array(color)
    if ((color.shape == (len(xyz), 3)) | (color.shape == (len(xyz), 4))):
        # then we have explicit colors
        if color.dtype != np.uint8:
            # if not passing uint8 assume 0-1 mapping
            assert(np.max(color) <= 1.0)
            assert(np.min(color) >= 0)
            color = np.uint8(color*255)
    elif color.shape == (len(xyz),):
        # then we want to map colors
        map_colors = True
    elif color.shape == (3,):
        # then we have one explicit color
        assert(np.max(color)<=1.0)
        assert(np.min(color)>=0)
        car = np.array(color*255, dtype=np.uint8) 
        color = np.repeat(car[np.newaxis,:],len(xyz),axis=0)
    else:
        raise ValueError(
            'color must have shapse Nx3 if explicitly setting, or (N,) if mapping, or (3,)')
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
    """ function for producing a vtkActor from a trimesh_io.Mesh

    Parameters
    ----------
    mesh : trimesh_io.Mesh
        a mesh to visualize
    color: various
        a len 3 iterator of a solid color to label mesh
        overridden by vertex_colors if passed
    opacity: float
        the opacity of the mesh (default .1)
    vertex_colors: np.array
        a np.array Nx3 list of explicit colors  (where N is len(mesh.vertices))
        OR
        a np.array of len(N) list of values to map through a colormap
        default (None) will use color to color mesh
    face_colors: np.array
        a np.array of Mx3 list of explicit colors (where M is the len(mesh.faces))
        OR
        a np.array of len(M) list of values to map through a colormap
        (default None will use color for mesh)
    lut: np.array
        not implemented
    calc_normals: bool
        whether to calculate normals on the mesh.  Default (True)
        will take more time, but will render a smoother mesh
        not compatible with sbow_link_edges. default True
    show_link_edges: bool
        whether to show the link_edges as lines. Will prevent calc_normals.
        default False
    line_width: int
        how thick to show lines (default 3)

    Returns
    -------
    vtk.vtkActor
        vtkActor representing the mesh (to be passed to render_actors)

    """
    if show_link_edges:
        mesh_poly = trimesh_to_vtk(mesh.vertices, mesh.faces, mesh.link_edges)
    else:
        mesh_poly = trimesh_to_vtk(mesh.vertices, mesh.faces, None)
    if vertex_colors is not None:
        vertex_color, map_vertex_color = process_colors(
            vertex_colors, mesh.vertices)
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
    """
    function to make a vtkActor from a skeleton class with different coloring options

    Parameters
    ----------
    sk : skeleton.Skeleton
        the skeleton class to create a render
    edge_property: str
        the key to the edge_properties dictionary on the sk object to use for coloring
        default None .. use color instead
    vertex_property: str
        the key to the vertex_properteis dictionary on the sk object to use for coloring
        default NOne ... use color instead
    vertex_data: np.array
        what data to color skeleton vertices by
        default None... use color intead
    normalize_property: bool
        whether to normalize the property data (edge/vertex) with dividing by np.nanmax
    color: tuple
        a 3 tuple in the [0,1] range of the color of the skeletoni
    line_width: int
        the width of the skeleton (default 3)
    opacity: float
        the opacity [0,1] of the mesh (1 = opaque, 0 = invisible)
    lut_map: np.array
        not implemented

    Returns
    -------
    vtk.vtkActor
        actor representing the skeleton

    """
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
        colors, map_colors = process_colors(data, sk.vertices)
        vtk_colors = numpy_to_vtk(colors)
        vtk_colors.SetName('colors')   
        sk_mesh.GetPointData().SetScalars(vtk_colors)
        if map_colors:
            lut = vtk.vtkLookupTable()
            if lut_map is not None:
                lut_map(lut)
            lut.Build()
            mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(line_width)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)
    return actor


def point_cloud_actor(xyz,
                      size=100,
                      color=(0,0,0),
                      opacity=1):
    """function to make a vtk.vtkActor from a set of xyz points that renders them as spheres

    Parameters
    ----------
    xyz : np.array
        a Nx3 array of points
    size: float or np.array
        the size of each of the points, or a N long array of sizes of each point
    color: len(3) iterator or np.array
        the color of all the points, or the color of each point individually as a N long array
        or a Nx3 list of explicit colors [0,1] range
    opacity: float
        the [0,1] opacity of mesh

    Returns
    -------
    vtk.vtkActor
        an actor with each of the xyz points as spheres of the specified size and color

    """
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
        raise ValueError(
            'Size must be either a scalar or an len(xyz) x 1 array')
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
    actor.GetProperty().SetOpacity(opacity)
    return actor


def linked_point_actor(vertices_a, vertices_b,
                       inds_a=None, inds_b=None,
                       line_width=1, color=(0, 0, 0), opacity=0.2):
    """ function for making polydata with lines between pairs of points

    Parameters
    ----------
    vertices_a : np.array
        a Nx3 array of point locations in xyz
    vertices_b : np.array
        a Nx3 array of point locations in xyz
    inds_a: np.array
        the indices in vertices_a to use (default None is all of them)
    inds_b: np.array
        the indices in vertices_b to use (default None is all of them)
    line_width : int
        the width of lines to draw (default 1)
    color : iterator
        a len(3) iterator (tuple, list, np.array) with the color [0,1] to use
    opacity: float
         a [0,1] opacity to render the lines

    Returns
    -------
    vtk.vtkActor
        an actor representing the lines between the points given with the color and opacity
        specified. To be passed to render_actors

    """
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


def oriented_camera(center, up_vector=(0, -1, 0), backoff=500, backoff_vector=(0, 0, 1)):
    '''
    Generate a camera pointed at a specific location, oriented with a given up
    direction, set to a backoff of the center a fixed distance with a particular direction

    Parameters
    ----------
    center : iterator
        a len 3 iterator (tuple, list, np.array) with the x,y,z location of the camera's focus point
    up_vector: iterator
        a len 3 iterator (tuple, list, np.array) with the dx,dy,dz direction of the camera's up direction
        default (0,-1,0) negative y is up.
    backoff: float
        distance in global space for the camera to be moved backward from the center point (default 500)
    backoff_vector: iterator
        a len 3 iterator (tuple, list, np.array) with the dx,dy,dz direction to back camera off of the focus point

    Returns
    -------
    vtk.vtkCamera
        the camera object representing the desired camera location, orientation and focus parameters

    '''
    camera = vtk.vtkCamera()

    pt_center = center

    vup = np.array(up_vector)
    vup = vup/np.linalg.norm(vup)

    bv = np.array(backoff_vector)
    pt_backoff = pt_center - backoff * 1000 * bv

    camera.SetFocalPoint(*pt_center)
    camera.SetViewUp(*vup)
    camera.SetPosition(*pt_backoff)
    return camera


def render_actors_360(actors, directory, nframes, camera_start=None, start_frame=0,
                      video_width=1280, video_height=720, scale=4, do_save=True, back_color=(1, 1, 1)):
    """
    Function to create a series of png frames which rotates around
    the Azimuth angle of a starting camera
    This will save images as a series of png images in the directory
    specified.
    The movie will start at time 0 and will go to frame nframes,
    completing a 360 degree rotation in that many frames.
    Keep in mind that typical movies are encoded at 15-30
    frames per second and nframes is units of frames.

    Parameters
    ----------
    actors :  list of vtkActor's
        list of vtkActors to render
    directory : str
        folder to save images into
    nframes : int
        number of frames to render
    camera_start : vtk.Camera
        camera to start rotation, default=None will fit actors in scene
    start_frame : int
        number to save the first frame number as... (default 0)
        i.e. frames will start_frame = 5, first file would be 005.png
    video_width : int
        size of video in pixels
    video_height : int
        size of the video in pixels
    scale : int
        how much to expand the image
    do_save : bool
        whether to save the images to disk or just play interactively
    back_color : iterable
        a len(3) iterable with the background color [0,1]rgb
    Returns
    -------
    vtkRenderer
        the renderer used to render
    endframe
        the last frame written
    Example
    -------
    ::

        from meshparty import trimesh_io, trimesh_vtk
        mm = trimesh_io.MeshMeta(disk_cache_path = 'meshes')
        mesh = mm.mesh(filename='mymesh.obj')
        mesh_actor = trimesh_vtk.mesh_actor(mesh)
        mesh_center = np.mean(mesh.vertices, axis=0)
        camera_start = trimesh_vtk.oriented_camera(mesh_center)

        render_actors_360([mesh_actor], 'movie', 360, camera_start=camera_start)
    """
    print('starting')
    if camera_start is None:
        frame_0_file = os.path.join(directory, "0000.png")
        ren = render_actors(actors,
                            do_save=True,
                            filename=frame_0_file,
                            video_width=video_width,
                            video_height=video_height,
                            back_color=back_color)
        print('done rendering')
        camera_start = ren.GetActiveCamera()
    print('camera_start done')
    cameras = []
    times = []
    for k, angle in enumerate(np.linspace(0, 360, nframes)):
        angle_cam = vtk.vtkCamera()
        angle_cam.ShallowCopy(camera_start)
        angle_cam.SetParallelProjection(camera_start.GetParallelProjection())
        angle_cam.SetParallelScale(camera_start.GetParallelScale())
        angle_cam.Azimuth(angle)
        cameras.append(angle_cam)
        times.append(k)
    print('cameras ready')
    return render_movie(actors, directory,
                        times=times,
                        cameras=cameras,
                        video_height=video_height,
                        video_width=video_width,
                        scale=scale,
                        do_save=do_save,
                        start_frame=start_frame,
                        back_color=back_color)


def _setup_renderer(video_width, video_height, back_color, camera=None):
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(video_width, video_height)
    ren.SetBackground(*back_color)
    ren.UseFXAAOn()
    # ren.SetBackground( 1, 1, 1)
    if camera is not None:
        ren.SetActiveCamera(camera)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    return ren, renWin, iren

def make_camera_interpolator(times, cameras, linear=False):
    assert(len(times) == len(cameras))
    camera_interp = vtk.vtkCameraInterpolator()
    for t, cam in zip(times, cameras):
        camera_interp.AddCamera(t, cam)
    if linear:
        camera_interp.SetInterpolationTypeToLinear()
    return camera_interp

def render_movie(actors, directory, times, cameras, start_frame=0,
                 video_width=1280, video_height=720, scale=4,
                 do_save=True, back_color=(1, 1, 1), linear=False):
    """
    Function to create a series of png frames based upon a defining 
    a set of cameras at a set of times.
    This will save images as a series of png images in the directory
    specified.
    The movie will start at time 0 and will go to frame np.max(times)
    Reccomend to make times start at 0 and the length of the movie
    you want.  Keep in mind that typical movies are encoded at 15-30
    frames per second and times is units of frames.

    Parameters
    ----------
    actors :  list of vtkActor's
        list of vtkActors to render
    directory : str
        folder to save images into
    times : np.array
        array of K frame times to set the camera to
    cameras : list of vtkCamera's
        array of K vtkCamera objects. movie with have cameras[k]
        at times[k]. 
    start_frame : int
        number to save the first frame number as... (default 0)
        i.e. frames will start_frame = 5, first file would be 005.png
    video_width : int
        size of video in pixels
    video_height : int
        size of the video in pixels
    scale : int
        how much to expand the image
    do_save : bool
        whether to save the images to disk or just play interactively
    Returns
    -------
    vtkRenderer
        the renderer used to render
    endframe
        the last frame written
    Example
    -------
    ::

        from meshparty import trimesh_io, trimesh_vtk
        mm = trimesh_io.MeshMeta(disk_cache_path = 'meshes')
        mesh = mm.mesh(filename='mymesh.obj')
        mesh_actor = trimesh_vtk.mesh_actor(mesh)
        mesh_center = np.mean(mesh.vertices, axis=0)

        camera_start = trimesh_vtk.oriented_camera(mesh_center, backoff = 10000, backoff_vector=(0, 0, 1))
        camera_180 = trimesh_vtk.oriented_camera(mesh_center, backoff = 10000, backoff_vector=(0, 0, -1))
        times = np.array([0, 90, 180])
        cameras = [camera_start, camera_180, camera_start]
        render_movie([mesh_actor],
                'movie',
                times,
                cameras)
    """
    camera_interp=make_camera_interpolator(times, cameras, linear=linear)

    def interpolate_camera(actors, camera, t):
        camera_interp.InterpolateCamera(t, camera)
    
    renWin, end_frame = render_movie_flexible(actors, directory, times,
                                          interpolate_camera,
                                          start_frame=start_frame,
                                          video_width=video_width,
                                          video_height=video_height,
                                          scale=scale,
                                          do_save=do_save,
                                          back_color=back_color)
    return renWin, end_frame

    
def render_movie_flexible(actors, directory, times, frame_change_function, start_frame=0,
                 video_width=1280, video_height=720, scale=4, camera=None,
                 do_save=True, back_color=(1, 1, 1)):
    """
    Function to create a series of png frames based upon a defining 
    a frame change function that will alter actors and camera at
    each time point
    This will save images as a series of png images in the directory
    specified.
    The movie will start at time 0 and will go to frame np.max(times)
    Reccomend to make times start at 0 and the length of the movie
    you want.  Keep in mind that typical movies are encoded at 15-30
    frames per second and times is units of frames.

    This is the most general of the movie making functions,
    and can be used to custom change coloring of actors or their positions
    over time using tranformations. 

    Parameters
    ----------
    actors :  list of vtkActor's
        list of vtkActors to render
    directory : str
        folder to save images into
    times : np.array
        array of K frame times to set the camera to
    frame_change_function : func
        a function that takes (actors, camera, t) as arguments.
        where actors are the list of actors passed here,
        camera is the camera for the rendering,
        and t is the current frame number.
        This function may alter the actors and camera as a function
        of time in some user defined manner.
    start_frame : int
        number to save the first frame number as... (default 0)
        i.e. frames will start_frame = 5, first file would be 005.png
    video_width : int
        size of video in pixels
    video_height : int
        size of the video in pixels
    scale : int
        how much to expand the image
    do_save : bool
        whether to save the images to disk or just play interactively
    Returns
    -------
    vtkRenderer
        the renderer used to render
    endframe
        the last frame written
    Example
    -------
    ::

        from meshparty import trimesh_io, trimesh_vtk
        mm = trimesh_io.MeshMeta(disk_cache_path = 'meshes')
        mesh = mm.mesh(filename='mymesh.obj')
        mesh_actor = trimesh_vtk.mesh_actor(mesh)
        mesh_center = np.mean(mesh.vertices, axis=0)

        camera_start = trimesh_vtk.oriented_camera(mesh_center, backoff = 10000, backoff_vector=(0, 0, 1))
        camera_180 = trimesh_vtk.oriented_camera(mesh_center, backoff = 10000, backoff_vector=(0, 0, -1))
        times = np.array([0, 90, 180])
        cameras = [camera_start, camera_180, camera_start]
        render_movie([mesh_actor],
                'movie',
                times,
                cameras)
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


    if camera is None:
        camera = vtk.vtkCamera()
    # create a rendering window and renderer
    ren, renWin, iren = _setup_renderer(
        video_width, video_height, back_color, camera=camera)

    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a)

    imageFilter = vtk.vtkWindowToImageFilter()
    imageFilter.SetInput(renWin)
    imageFilter.SetScale(scale)
    imageFilter.SetInputBufferTypeToRGB()
    imageFilter.ReadFrontBufferOff()
    imageFilter.Update()

    # Setup movie writer
    if do_save:
        moviewriter = vtk.vtkPNGWriter()
        moviewriter.SetInputConnection(imageFilter.GetOutputPort())
        renWin.OffScreenRenderingOn()

    for i in np.arange(0, np.max(times)+1):
        frame_change_function(actors, camera, i)  
        ren.ResetCameraClippingRange()
        camera.ViewingRaysModified()
        renWin.Render()

        if do_save:
            filename = os.path.join(directory, "%04d.png" % (i+start_frame))
            moviewriter.SetFileName(filename)
            # Export a current frame
            imageFilter.Update()
            imageFilter.Modified()
            moviewriter.Write()
    
    renWin.Finalize()
    return renWin, i+start_frame

def scale_bar_actor(center, camera, length=10000, color=(0, 0, 0), linewidth=5, font_size=20):
    """Creates a xyz 3d scale bar actor located at a specific location with a given size

    Parameters
    ----------
    center : iterable
        a length 3 iterable of xyz position
    camera : vtk.vtkCamera
        the camera the scale bar should follow
    length : int, optional
        length of each of the xyz axis, by default 10000
    color : tuple, optional
        color of text and lines, by default (0,0,0)
    linewidth : int, optional
        width of line in pixels, by default 5
    font_size : int, optional
        font size of xyz labels, by default 20

    Returns
    -------
    vtk.vktActor
        scale bar actor to add to render_actors

    """
    axes_actor = vtk.vtkCubeAxesActor2D()
    axes_actor.SetBounds(center[0], center[0]+length,
                         center[1], center[1]+length,
                         center[2], center[2]+length)
    # this means no real labels
    axes_actor.SetLabelFormat("")
    axes_actor.SetCamera(camera)
    # this turns off the tick marks and labelled numbers
    axes_actor.SetNumberOfLabels(0)
    # this affects whether the corner of the 3 axis
    # changes as you rotate the view
    # this option makes it stay constant
    axes_actor.SetFlyModeToNone()
    axes_actor.SetFontFactor(1.0)
    axes_actor.GetProperty().SetColor(*color)
    axes_actor.GetProperty().SetLineWidth(linewidth)
    # this controls the color of text
    tprop = vtk.vtkTextProperty()
    tprop.SetColor(*color)
    # no shadows on text
    tprop.ShadowOff()
    tprop.SetFontSize(font_size)
    # makes the xyz and labels the same
    axes_actor.SetAxisTitleTextProperty(tprop)
    axes_actor.SetAxisLabelTextProperty(tprop)

    return axes_actor


def values_to_colors(values, cmap, vmin=None, vmax=None):
    """
    Function to map a set of values through a colormap
    to get RGB values in order to facilitate coloring of meshes.

    Parameters
    ----------
    values: array-like, (n_vertices, )
        values to pass through colormap
    cmap: array-like, (n_colors, 3)
        colormap describing the RGB values from vmin to vmax
    vmin : float 
        (optional) value that should receive minimum of colormap.
        default to minimum of values
    vmax : float 
        (optional) values that should receive maximum of colormap
        default to maximum of values

    Output
    ------
    colors: array-like,  (n_vertices, 3)
        RGB values for each entry in values (as np.uint8 [0-255])

    Example
    -------

    Assuming mesh object and 'values' have been calculated already 
    ::

        import seaborn as sns

        cmap = np.array(sns.color_palette('viridis', 1000))
        clrs = trimesh_vtk.values_to_colors(values, cmap)
        mesh_actor = trimesh_io.mesh_actor(mesh, vertex_colors = clrs, opacity=1.0)
        trimesh_vtk.render_actors([mesh_actor])

    """
    n_colors = cmap.shape[0]
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    values = np.clip(values, vmin, vmax)
    r = np.interp(x=values, xp=np.linspace(vmin, vmax, n_colors), fp=cmap[:, 0])
    g = np.interp(x=values, xp=np.linspace(vmin, vmax, n_colors), fp=cmap[:, 1])
    b = np.interp(x=values, xp=np.linspace(vmin, vmax, n_colors), fp=cmap[:, 2])
    colors = np.vstack([r, g, b]).T
    colors = (colors * 255).astype(np.uint8)
    return colors