import numpy as np


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
