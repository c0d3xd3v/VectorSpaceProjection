import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import igl

def compute_normal(v1, v2):
    # Calculate the normal vector using the cross product
    cross_product = np.cross(v1, v2)
    norm = np.linalg.norm(cross_product)
    if norm == 0:
        return np.array([0, 0, 1])  # Default normal if cross product is zero
    return cross_product / norm  # Normalize the normal vector


def compute_vertex_normals(vertices, triangles):
    # Convert vertices and triangles to numpy arrays, ensuring vertices are float64
    vertices = np.array(vertices, dtype=np.float64)
    triangles = np.array(triangles)

    # Initialize an array to store normals for each vertex
    vertex_normals = np.zeros_like(vertices)

    # Step 1: Compute face normals
    for triangle in triangles:
        v1, v2, v3 = vertices[triangle]
        edge1 = v2 - v1
        edge2 = v3 - v1
        face_normal = compute_normal(edge1, edge2)
        # face_normal = np.cross(edge1, edge2)
        # face_normal /= np.linalg.norm(face_normal)  # Normalize the face normal

        # Step 2: Add the face normal to each vertex of the triangle
        vertex_normals[triangle[0]] += face_normal
        vertex_normals[triangle[1]] += face_normal
        vertex_normals[triangle[2]] += face_normal

    # Step 3: Normalize the vertex normals
    vertex_normals = np.array([normal / np.linalg.norm(normal) for normal in vertex_normals])

    return vertex_normals


def compute_vertex_normals_with_sharp_edges(vertices, triangles, feature_angle_deg=30.0):
    """
    Compute vertex normals for a triangle mesh while respecting sharp edges based on a feature angle.
    
    :param vertices: (Nx3) array of vertex positions.
    :param triangles: (Mx3) array of triangle indices.
    :param feature_angle_deg: The angle in degrees above which edges are considered sharp.
    
    :return: (Nx3) array of vertex normals.
    """
    # Ensure inputs are NumPy arrays
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    feature_angle_rad = np.radians(feature_angle_deg)

    # Step 1: Compute face normals
    def compute_face_normals(vertices, triangles):
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        # Normalize face normals
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]
        return normals

    face_normals = compute_face_normals(vertices, triangles)

    # Step 2: Initialize vertex normals and adjacency dictionary
    vertex_normals = np.zeros_like(vertices)
    adjacency = {i: [] for i in range(len(vertices))}

    # Step 3: Build adjacency list for each vertex and its connected faces
    for i, tri in enumerate(triangles):
        for j in range(3):
            adjacency[tri[j]].append(i)

    # Step 4: Accumulate vertex normals
    for vertex_idx, adjacent_faces in adjacency.items():
        normal_sum = np.zeros(3)
        for i in range(len(adjacent_faces)):
            normal_i = face_normals[adjacent_faces[i]]
            for j in range(i + 1, len(adjacent_faces)):
                normal_j = face_normals[adjacent_faces[j]]
                # Compute angle between face normals
                angle = np.arccos(np.clip(np.dot(normal_i, normal_j), -1.0, 1.0))
                # Only accumulate normals if the angle is less than the feature angle (non-sharp edge)
                if angle <= feature_angle_rad:
                    normal_sum += normal_i
                    normal_sum += normal_j
        # Normalize the summed normal and assign it to the vertex
        if np.linalg.norm(normal_sum) > 0:
            vertex_normals[vertex_idx] = normal_sum / np.linalg.norm(normal_sum)

    return vertex_normals
