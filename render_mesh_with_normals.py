import sys
import vtk_
import geo
import fileio

import numpy as np


filepath = sys.argv[1]
vertices, triangles, nodeData = fileio.load_mesh_vtk_unstructured_grid(filepath)
vertex_normals = geo.compute_vertex_normals_with_sharp_edges(vertices, triangles)
vertex_normals_ = vtk_.compute_vertex_normals_with_sharp_edges(vertices, triangles)

colormap = fileio.create_vtk_color_transfer_function_from_xml('data/colormap.xml')
name = '(11319.71-0j)Hz'
mesh_actor = vtk_.create_mesh_actor(vertices, triangles, nodeData[name], colormap)
mesh_actor_ = vtk_.create_vtk_actor_with_vertex_normals(vertices, triangles, vertex_normals_)

vectorfield_actor = vtk_.create_vectorfield_actor(vertices, nodeData[name], colormap)
normalfield_actor = vtk_.create_vectorfield_actor(vertices, vertex_normals)

actors = [mesh_actor]
vtk_.render_actors(actors)
