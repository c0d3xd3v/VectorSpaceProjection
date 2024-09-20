import sys
import igl
import numpy as np
import vtk
import vtk_

def load_mesh_with_libigl(filepath):
    # Load the mesh using libigl
    V = igl.read_triangle_mesh(filepath)[0]  # Vertices
    F = igl.read_triangle_mesh(filepath)[1]  # Faces (Triangles)

    vertices = V.tolist()
    triangles = F.tolist()

    return vertices, triangles

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
        face_normal = np.cross(edge1, edge2)
        face_normal /= np.linalg.norm(face_normal)  # Normalize the face normal

        # Step 2: Add the face normal to each vertex of the triangle
        vertex_normals[triangle[0]] += face_normal
        vertex_normals[triangle[1]] += face_normal
        vertex_normals[triangle[2]] += face_normal

    # Step 3: Normalize the vertex normals
    vertex_normals = np.array([normal / np.linalg.norm(normal) for normal in vertex_normals])

    return vertex_normals


# Example usage:
# Lade ein Mesh (z.B. ein .obj- oder .off-File). Hier ist ein Beispielpfad:
filepath = sys.argv[1]  # Replace with your actual mesh file

# Mesh laden mit libigl
vertices, triangles = load_mesh_with_libigl(filepath)

# Compute the vertex normals using numpy
vertex_normals = compute_vertex_normals(vertices, triangles)

# Render the mesh with computed vertex normals visualized as arrows
mesh_actor, arrow_actor = vtk_.create_mesh_actor_with_normals(vertices, triangles, vertex_normals)
actors = [mesh_actor, arrow_actor]
vtk_.render_actors(actors)

