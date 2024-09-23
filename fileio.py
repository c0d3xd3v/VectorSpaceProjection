import igl
import vtk

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


def load_triangle_mesh(filepath):
    # Load the mesh using libigl
    V = igl.read_triangle_mesh(filepath)[0]  # Vertices
    F = igl.read_triangle_mesh(filepath)[1]  # Faces (Triangles)
    return V.tolist(), F.tolist()


def load_mesh_vtk_unstructured_grid(filepath):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.Update()

    unstructured_grid: vtk.vtkUnstructuredGrid = reader.GetOutput()
    points_vtk = unstructured_grid.GetPoints()

    triangles_vtk = []
    for i in range(unstructured_grid.GetNumberOfCells()):
        cell = unstructured_grid.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:  # Check if the cell is a triangle
            triangle = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            triangles_vtk.append(triangle)

    # Convert triangles list to a NumPy array
    triangles = np.array(triangles_vtk)
    vertices = vtk_to_numpy(points_vtk.GetData())


    point_data = unstructured_grid.GetPointData()
    num_point_data_arrays = point_data.GetNumberOfArrays()
    print(f"Number of PointData arrays: {num_point_data_arrays}")

    point_data_arrays = {}
    for i in range(num_point_data_arrays):
        array_name = point_data.GetArrayName(i)
        array = point_data.GetArray(i)
        numpy_array = vtk_to_numpy(array)
        
        num_components = array.GetNumberOfComponents()
        if num_components > 1:
            numpy_array = numpy_array.reshape(-1, num_components)
        else:
            numpy_array = numpy_array.flatten()
        point_data_arrays[array_name] = numpy_array

        data_type = "Scalar" if num_components == 1 else f"Vector (dim {num_components})" if num_components == 3 else f"Tensor (dim {num_components})"
        print(f"Stored {array_name} as {data_type} with shape {numpy_array.shape}")

    return vertices.tolist(), triangles.tolist(), point_data_arrays
