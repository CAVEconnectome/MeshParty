import pytest
import numpy as np
import vtk
from meshparty import trimesh_vtk, trimesh_io
from basic_test import basic_mesh, mesh_with_extra_edges

def test_vtk_cell_array(basic_mesh):
    '''
    Handles both 
    '''
    edge_array = basic_mesh.edges
    edge_cell_array = trimesh_vtk.numpy_to_vtk_cells(edge_array)
    assert isinstance(edge_cell_array, vtk.vtkCellArray)

    edges_back = trimesh_vtk.vtk_cellarray_to_shape(edge_cell_array,
                                                    len(basic_mesh.edges))
    assert np.all(edges_back == edge_array)

    edges_back = trimesh_vtk.vtk_cellarray_to_shape(edge_cell_array.GetData(),
                                                    len(basic_mesh.edges))
    assert np.all(edges_back == edge_array)



def test_graph_to_vtk(basic_mesh):
    vertices = basic_mesh.vertices
    edges = basic_mesh.edges

    vtk_poly = trimesh_vtk.graph_to_vtk(vertices, edges)
    assert isinstance(vtk_poly, vtk.vtkPolyData)

    points_back = trimesh_vtk.vtk_to_numpy(vtk_poly.GetPoints().GetData())
    cell_array_back = vtk_poly.GetLines()
    shapes_back = trimesh_vtk.vtk_cellarray_to_shape(cell_array_back.GetData(),len(edges))
    assert np.all(points_back==vertices)
    assert np.all(shapes_back==edges)
    
# def test_numpy_rep_to_vtk(mesh_with_extra_edges):
#     vertices_array = mesh_with_extra_edges.vertices
#     shape_array = mesh_with_extra_edges.edges
#     edge_array = mesh_with_extra_edges.mesh_edges
#     vtk_poly,vtk_cells,vtk_edges = trimesh_vtk.numpy_rep_to_vtk(
#                                 vertices_array,shape_array,edges=edge_array)

#     assert isinstance(vtk_poly, vtk.vtkPolyData)
#     assert isinstance(vtk_cells, vtk.vtkCellArray)
#     assert isinstance(vtk_edges, vtk.vtkCellArray)

#     #points_back,shapes_back,edges_back = trimesh_vtk.vtk_poly_to_mesh_components(vtk_poly)
#     points_back = trimesh_vtk.vtk_to_numpy(vtk_poly.GetPoints().GetData())
#     shapes_back = trimesh_vtk.vtk_cellarray_to_shape(vtk_cells.GetData(),
#                                                     len(mesh_with_extra_edges.edges))
#     edges_back = trimesh_vtk.vtk_cellarray_to_shape(vtk_edges.GetData(),
#                                                     len(mesh_with_extra_edges.edges))

#     assert np.all(points_back == vertices_array)
#     assert np.all(edges_back == edge_array)
#     assert np.all(shapes_back == shape_array)


    
    

