import sys
import vtk_
import geo
import fileio


filepath = sys.argv[1]
vertices, triangles, nodeData = fileio.load_mesh_vtk_unstructured_grid(filepath)
vertex_normals = geo.compute_vertex_normals(vertices, triangles)

colormap = fileio.create_vtk_color_transfer_function_from_xml('data/colormap.xml')
mesh_actor = vtk_.create_mesh_actor(vertices, triangles, nodeData['eigenmode5'], colormap)
vectorfield_actor = vtk_.create_vectorfield_actor(vertices, nodeData['eigenmode5'])
normalfield_actor = vtk_.create_vectorfield_actor(vertices, vertex_normals)

actors = [mesh_actor, vectorfield_actor, normalfield_actor]
vtk_.render_actors(actors)
