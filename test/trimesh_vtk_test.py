import pytest
import numpy as np
from meshparty import trimesh_vtk, trimesh_io
from basic_test import basic_mesh

def test_vtk_cell_array(basic_mesh):
    edge_array = basic_mesh.edges
    edge_cell_data = trimesh_vtk.numpy_to_vtk_cells(edge_array)
    assert isinstance(edge_cell_data, vtk.vtkCellArray)

    edges_back = trimesh_vtk.vtk_cellarray_to_shape(edge_cell_array.GetData(),
                                                    len(test_mesh.edges))

    assert np.all(edges_back == edge_array)
