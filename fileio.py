import igl


def load_mesh_with_libigl(filepath):
    # Load the mesh using libigl
    V = igl.read_triangle_mesh(filepath)[0]  # Vertices
    F = igl.read_triangle_mesh(filepath)[1]  # Faces (Triangles)

    vertices = V.tolist()
    triangles = F.tolist()

    return vertices, triangles