import vtk
import geo
import numpy as np

from vtkmodules.util.numpy_support import numpy_to_vtk

def render_actors(actors, camera_position=[1, 0, 0]):
    """
    renders each actor in the actors list, seen from 
    camera position.
    Args:
        actors (list): list of vtk actors
        camera_position (list, optional): poisition of the camera the scene is 
                                          viewed from. Defaults to [1, 0, 0].
    """
    # VTK Visualization
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # White background

    # Configure depth testing for better rendering of overlapping objects
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(4)
    renderer.SetOcclusionRatio(0.1)

    cam = renderer.GetActiveCamera()
    cam.SetViewUp(camera_position)

    # Customize actors' appearance
    for actor in actors:
        prop = actor.GetProperty()
        prop.SetAmbient(0.0)    # Slight ambient reflection for smoother shading
        prop.SetDiffuse(1.0)    # Strong diffuse reflection
        #prop.SetSpecular(0.2)   # Add some specular highlights
        #prop.SetSpecularPower(10)  # Control sharpness of specular highlights        
        prop.SetColor(0.85, 0.85, 0.85)
        renderer.AddActor(actor)

    #create_bounding_box_ground_plane(renderer)

    # Render window setup
    render_window = vtk.vtkRenderWindow()
    width, height = 800, 800
    render_window.SetSize(width, height)
    viewport_max_sizes = render_window.GetScreenSize()
    xwpos = int(viewport_max_sizes[0]/2 - width)
    ywpos = int(viewport_max_sizes[1]/2 - height)
    render_window.SetPosition(xwpos, ywpos)
    
    render_window.AddRenderer(renderer)

    # Interactor setup
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # Add lighting at camera
    create_light_source_at_camera(renderer, renderer.GetActiveCamera())

    # Shadows for enhanced realism
    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()

    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)

    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)
    renderer.SetPass(cameraP)

    render_window.SetMultiSamples(2)  # Anti-aliasing for smoother edges
    render_window.Render()
    renderer.ResetCamera()

    render_window.Render()
    render_window_interactor.Start()


def create_light_source_at_camera(renderer, camera, offset=[0.1, 0.1, 0.1]):
    """Creates a light source and positions it at the camera's position."""
    colors = vtk.vtkNamedColors()

    # Using a warmer white light for more natural look
    colors.SetColor('WarmLight', [255, 244, 229, 255])  # Soft warm light
    colors.SetColor('CoolLight', [200, 200, 255, 255])  # Cooler, bluish light

    light = vtk.vtkLight()
    light.SetFocalPoint(0, 0, 0)
    light_position = np.array(camera.GetPosition()) + np.array(offset)
    light.SetPosition(light_position[0], light_position[1], light_position[2])
    light.SetColor(colors.GetColor3d('WarmLight'))  # Warm light for a nice contrast
    light.SetIntensity(1.0)  # Slightly stronger light
    renderer.AddLight(light)

    # Adjust light position dynamically if the camera moves
    def update_light_position(*args):
        light_position = np.array(camera.GetPosition()) + np.array(offset)
        light.SetPosition(light_position[0], light_position[1], light_position[2])
        light.Modified()

    camera.AddObserver('ModifiedEvent', update_light_position)

    return light, update_light_position


def create_bounding_box_ground_plane(renderer, normal_vector=[1, 0, 0]):
    """
    Creates a ground plane based on the bounding box of the scene's actors,
    centered under all actors with a prescribed normal vector.
    
    Parameters:
    renderer : vtkRenderer
        The renderer containing all the actors.
    normal_vector : list or array-like
        A 3-element list defining the normal vector of the plane.
    """
    # Initialize bounds for the entire scene (actors)
    bounds = [float('inf'), -float('inf'), float('inf'), -float('inf'), 
              float('inf'), -float('inf')]

    # Iterate over all actors to get the collective bounding box
    actors = renderer.GetActors()
    actors.InitTraversal()

    for i in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        actor_bounds = actor.GetBounds()

        # Update the global bounds based on actor bounds
        bounds[0] = min(bounds[0], actor_bounds[0])  # X min
        bounds[1] = max(bounds[1], actor_bounds[1])  # X max
        bounds[2] = min(bounds[2], actor_bounds[2])  # Y min
        bounds[3] = max(bounds[3], actor_bounds[3])  # Y max
        bounds[4] = min(bounds[4], actor_bounds[4])  # Z min (ground)
        bounds[5] = max(bounds[5], actor_bounds[5])  # Z max

    offset = 0.1  # A small offset to lower the plane slightly under the actors

    # Calculate the center of the bounding box in the X and Y directions
    center_x = (bounds[0] + bounds[1]) / 2.0  # X center
    center_y = (bounds[2] + bounds[3]) / 2.0  # Y center
    z_min = bounds[4]  # Use the minimum Z bound for the plane's height

    # Calculate the width and height of the plane (spanning X and Y dimensions)
    width = bounds[1] - bounds[0]  # Width of the bounding box in X
    height = bounds[3] - bounds[2]  # Height of the bounding box in Y

    # Create a plane at the center of the bounding box
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(center_x - width / 2.0, center_y - height / 2.0, z_min - offset)
    plane.SetPoint1(center_x + width / 2.0, center_y - height / 2.0, z_min - offset)
    plane.SetPoint2(center_x - width / 2.0, center_y + height / 2.0, z_min - offset)

    # Set the prescribed normal vector
    plane.SetNormal(normal_vector[0], normal_vector[1], normal_vector[2])

    plane.SetResolution(10, 10)  # Set resolution for finer mesh

    # Create a mapper and actor for the plane
    plane_mapper = vtk.vtkPolyDataMapper()
    plane_mapper.SetInputConnection(plane.GetOutputPort())

    plane_actor = vtk.vtkActor()
    plane_actor.SetMapper(plane_mapper)

    # Customize the appearance of the plane
    plane_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # White color
    plane_actor.GetProperty().SetAmbient(0.7)          # High ambient reflection
    plane_actor.GetProperty().SetDiffuse(0.9)          # Strong diffuse reflection
    #plane_actor.GetProperty().SetOpacity(0.5)          # 50% transparency

    # Add the plane to the renderer
    renderer.AddActor(plane_actor)


def create_arrow_actor(start_point, end_point, color):
    # Erstellen der Pfeilquelle
    arrow_source = vtk.vtkArrowSource()
    arrow_source.SetTipResolution(64)
    arrow_source.SetShaftResolution(64)

    # Berechnen der Richtung des Pfeils
    direction = np.array(end_point) - np.array(start_point)
    length = np.linalg.norm(direction)
    direction /= length

    # Standardrichtung des Pfeils (von (0, 0, 0) nach (1, 0, 0))
    arrow_direction = np.array([1, 0, 0])

    # Berechne die Rotationsachse und den Winkel zwischen der Standardrichtung und der gewünschten Richtung
    axis = np.cross(arrow_direction, direction)
    axis_length = np.linalg.norm(axis)

    if axis_length != 0:
        axis /= axis_length  # Normalisieren der Achse
        angle = np.degrees(np.arcsin(axis_length))  # Winkel zwischen den Vektoren
    else:
        angle = 0  # Kein Winkel notwendig, wenn die Richtungen übereinstimmen

    # Transformieren des Pfeils
    transform = vtk.vtkTransform()
    transform.Translate(start_point)

    # Rotation anwenden, wenn der Winkel ungleich 0 ist
    if angle != 0:
        transform.RotateWXYZ(angle, axis)

    # Skalierung des Pfeils auf die richtige Länge
    transform.Scale(length, length, length)

    # Transformfilter anwenden
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(arrow_source.GetOutputPort())
    transform_filter.SetTransform(transform)

    # Mapper erstellen
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())

    # Actor erstellen
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    return actor


def create_line_actor(start_point, end_point, color):
    # Erstellen der Linie
    line_source = vtk.vtkLineSource()
    line_source.SetPoint1(start_point)
    line_source.SetPoint2(end_point)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    return actor


def create_plane_actor(n, point, color):
    normal = n
    normal_length = np.linalg.norm(normal)
    if normal_length == 0:
        normal = np.array([0, 0, 1])
    else:
        normal /= normal_length

    plane_source = vtk.vtkPlaneSource()
    plane_source.SetNormal(normal)
    plane_source.SetCenter(point)
    plane_source.SetXResolution(5)
    plane_source.SetYResolution(5)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    #actor.GetProperty().SetOpacity(0.9)
    return actor


def create_mesh_actor(vertices, triangles, nodeData=None, colormap=None):
    # Create a vtkPoints object and set the points from the vertices
    points = vtk.vtkPoints()
    for vertex in vertices:
        points.InsertNextPoint(vertex)

    # Create a vtkCellArray object to store the triangles
    triangles_vtk = vtk.vtkCellArray()
    for triangle in triangles:
        triangle_vtk = vtk.vtkTriangle()
        triangle_vtk.GetPointIds().SetId(0, triangle[0])
        triangle_vtk.GetPointIds().SetId(1, triangle[1])
        triangle_vtk.GetPointIds().SetId(2, triangle[2])
        triangles_vtk.InsertNextCell(triangle_vtk)

    # Create a vtkPolyData object to store the mesh
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(triangles_vtk)

    # Create a mapper and actor for the mesh
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(poly_data)

    if colormap != None and nodeData.all() != None:
        max_norms = np.max(np.abs(nodeData), axis=1)
        global_max_norm = np.max(max_norms)
        points_array = numpy_to_vtk(nodeData, deep=True)
        points_array.SetName("array")
        poly_data.GetPointData().AddArray(points_array)
        
        mesh_mapper.ScalarVisibilityOn()
        mesh_mapper.SetScalarModeToUsePointFieldData()
        mesh_mapper.InterpolateScalarsBeforeMappingOn()
        mesh_mapper.SetScalarRange([0, global_max_norm])
        mesh_mapper.SelectColorArray("array")
        mesh_mapper.SetLookupTable(colormap)

    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)

    return mesh_actor


def create_vectorfield_actor(vertices, vector_data, colormap=None, scale=0.1):
    # Create a vtkArrowSource to represent normals as arrows
    arrow_source = vtk.vtkArrowSource()
    arrow_source.SetTipResolution(16)
    arrow_source.SetShaftResolution(16)

    # Create a vtkPoints object to store the arrow positions (same as vertices)
    normal_points = vtk.vtkPoints()
    normal_vectors = vtk.vtkDoubleArray()
    normal_vectors.SetNumberOfComponents(3)  # 3D vectors

    # Store the magnitudes of the vectors for colormap
    magnitudes = vtk.vtkDoubleArray()
    magnitudes.SetName("Magnitudes")
    magnitudes.SetNumberOfComponents(1)

    for i in range(len(vertices)):
        normal_points.InsertNextPoint(vertices[i])
        normal_vectors.InsertNextTuple(vector_data[i])
        # Calculate and store the magnitude for coloring
        magnitudes.InsertNextValue(np.linalg.norm(vector_data[i]))

    # Create polydata for the glyphs (normals visualization)
    normals_polydata = vtk.vtkPolyData()
    normals_polydata.SetPoints(normal_points)
    normals_polydata.GetPointData().SetVectors(normal_vectors)
    # Add the magnitudes as scalar data
    normals_polydata.GetPointData().SetScalars(magnitudes)

    # Use vtkGlyph3D to generate arrows at each vertex position along the normal
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow_source.GetOutputPort())
    glyph.SetInputData(normals_polydata)
    glyph.SetVectorModeToUseVector()  # Use the normals stored in the polydata
    glyph.OrientOn()  # Align arrows with normals
    glyph.SetScaleFactor(scale)  # Scale the arrows
    glyph.ScalingOn()
    glyph.SetScaleModeToDataScalingOff()

    # Create a mapper and actor for the arrows (normals)
    arrow_mapper = vtk.vtkPolyDataMapper()
    arrow_mapper.SetInputConnection(glyph.GetOutputPort())

    # If colormap and vector data is provided
    if colormap is not None and vector_data is not None:
        # Turn on scalar visibility to use magnitudes for coloring
        arrow_mapper.ScalarVisibilityOn()
        arrow_mapper.SetScalarModeToUsePointData()  # Use the scalars for coloring
        arrow_mapper.SetScalarRange(magnitudes.GetRange())  # Set the range for colormap
        arrow_mapper.SetLookupTable(colormap)  # Apply colormap
    else:
        arrow_mapper.ScalarVisibilityOff()

    arrow_actor = vtk.vtkActor()
    arrow_actor.SetMapper(arrow_mapper)

    return arrow_actor


def compute_vertex_normals_with_sharp_edges(vertices, triangles, feature_angle=30.0):
    """
    Compute vertex normals with respect to sharp edges.
    
    :param vertices: A numpy array of shape (n_vertices, 3) representing the vertex positions.
    :param triangles: A numpy array of shape (n_triangles, 3) representing the triangle vertex indices.
    :param feature_angle: The angle threshold (in degrees) to detect sharp edges.
    :return: A numpy array of shape (n_vertices, 3) containing the computed vertex normals.
    """
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    # Convert feature angle to radians
    feature_angle_rad = np.radians(feature_angle)

    # Compute face normals
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    # Create a dictionary to store the normals per vertex
    vertex_normals = {i: [] for i in range(len(vertices))}

    # Associate face normals with each vertex
    for i, triangle in enumerate(triangles):
        for vertex_index in triangle:
            vertex_normals[vertex_index].append(face_normals[i])

    # Compute the final vertex normals, splitting normals at sharp edges
    final_normals = np.zeros_like(vertices)

    for vertex_index, normals in vertex_normals.items():
        # Convert the list of normals to a numpy array
        normals = np.array(normals)

        # Compute pairwise angles between normals
        angles = np.arccos(np.clip(np.dot(normals, normals.T), -1.0, 1.0))

        # Average normals that are within the feature angle threshold
        avg_normal = np.zeros(3)
        for normal in normals:
            within_angle = np.all(angles <= feature_angle_rad, axis=1)
            if np.any(within_angle):
                avg_normal += normal
        avg_normal /= np.linalg.norm(avg_normal)
        
        final_normals[vertex_index] = avg_normal

    return final_normals


def create_vtk_actor_with_vertex_normals(vertices, triangles, vertex_normals):
    """
    Create a VTK actor for the mesh with per-vertex normals.
    """
    # Create vtkPoints object
    points = vtk.vtkPoints()
    for vertex in vertices:
        points.InsertNextPoint(vertex)

    # Create vtkCellArray for triangles
    triangles_vtk = vtk.vtkCellArray()
    for triangle in triangles:
        triangle_vtk = vtk.vtkTriangle()
        triangle_vtk.GetPointIds().SetId(0, triangle[0])
        triangle_vtk.GetPointIds().SetId(1, triangle[1])
        triangle_vtk.GetPointIds().SetId(2, triangle[2])
        triangles_vtk.InsertNextCell(triangle_vtk)

    # Create vtkPolyData object to store the mesh
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(triangles_vtk)

    # Create vtkFloatArray for normals
    normals_array = vtk.vtkFloatArray()
    normals_array.SetNumberOfComponents(3)
    normals_array.SetName("Normals")

    # Insert vertex normals into the vtkFloatArray
    for normal in vertex_normals:
        normals_array.InsertNextTuple(normal)

    # Assign the vertex normals to the poly data
    poly_data.GetPointData().SetNormals(normals_array)

    # Create the mapper and actor
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(poly_data)

    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)

    return mesh_actor
