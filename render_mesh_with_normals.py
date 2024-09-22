import sys
import vtk_
import geo
import fileio

filepath = sys.argv[1]
vertices, triangles = fileio.load_mesh_with_libigl(filepath)
vertex_normals = geo.compute_vertex_normals(vertices, triangles)

mesh_actor, arrow_actor = vtk_.create_mesh_and_vectorfield_actors(vertices, triangles, vertex_normals)
actors = [mesh_actor, arrow_actor]
vtk_.render_actors(actors)
