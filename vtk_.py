import vtk
import geo
import numpy as np

def render_actors(actors, camera_position=[1, 0, 0]):
    # VTK Visualization
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1,1,1)  # Hintergrundfarbe auf weiß setzen
    # Konfiguriere Depth Testing
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(4)
    renderer.SetOcclusionRatio(0.1)

    cam = renderer.GetActiveCamera()
    cam.SetViewUp(camera_position)

    for actor in actors:
        actor.GetProperty().SetAmbient(0.7)
        actor.GetProperty().SetDiffuse(0.5)
        renderer.AddActor(actor)

    # Renderer-Instanz erstellen
    render_window = vtk.vtkRenderWindow()
    width, height = 800, 800
    render_window.SetSize(width, height)
    viewport_max_sizes = render_window.GetScreenSize()
    render_window.SetPosition(int(viewport_max_sizes[0]/2 - width), int(viewport_max_sizes[1]/2 - height))
    print(viewport_max_sizes)
    render_window.AddRenderer(renderer)

    # RenderWindowInteractor erstellen
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    light, update_light_position = create_light_source_at_camera(renderer, renderer.GetActiveCamera())

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()

    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)

    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)
    
    renderer.SetPass(cameraP)

    render_window.SetMultiSamples(8)
    render_window.Render()
    renderer.ResetCamera()

    render_window.Render()
    render_window_interactor.Start()


def create_light_source_at_camera(renderer, camera):
    """Creates a light source and positions it at the camera's position."""

    colors = vtk.vtkNamedColors()
    colors.SetColor('HighNoonSun', [255, 255, 251, 255])  # Color temp. 5400°K
    colors.SetColor('100W Tungsten', [255, 214, 170, 255])  # Color temp. 2850°K

    light = vtk.vtkLight()
    light.SetFocalPoint(0, 0, 0)
    light.SetPosition(0.0, -3.0, 0.0)
    light.SetColor(colors.GetColor3d('100W Tungsten'))
    light.SetIntensity(0.25)
    renderer.AddLight(light)

    def update_light_position(*args):
        light_position = camera.GetPosition()
        light_position += np.array([0.1, 0.1, 0.1])
        light.SetPosition(light_position[0], light_position[1], light_position[2])
        renderer.GetRenderWindow().Render()  # Re-render to apply the changes

    camera.AddObserver('ModifiedEvent', update_light_position)

    return light, update_light_position


def create_bounding_box_ground_plane(renderer):
    # Initialize bounds for the entire scene (actors)
    bounds = [float('inf'), -float('inf'), float('inf'), -float('inf'), float('inf'), -float('inf')]

    # Iterate over all actors to get the collective bounding box
    actors = renderer.GetActors()
    actors.InitTraversal()

    for i in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        actor_bounds = actor.GetBounds()

        s = 2.0
        # Update the global bounds
        bounds[0] = min(bounds[0], s*actor_bounds[0])  # X min
        bounds[1] = max(bounds[1], s*actor_bounds[1])  # X max
        bounds[2] = min(bounds[2], s*actor_bounds[2])  # Y min
        bounds[3] = max(bounds[3], s*actor_bounds[3])  # Y max
        bounds[4] = min(bounds[4], s*actor_bounds[4])  # Z min (ground)
        bounds[5] = max(bounds[5], s*actor_bounds[5])  # Z max

    offset = 0.1

    # Create a plane at the Z-min (bottom of the bounding box)
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(bounds[0], bounds[2], bounds[4] - offset)  # Bottom-left corner at Z-min
    plane.SetPoint1(bounds[1], bounds[2], bounds[4] - offset)  # Bottom-right corner at Z-min
    plane.SetPoint2(bounds[0], bounds[3], bounds[4] - offset)  # Top-left corner at Z-min
    plane.SetResolution(10, 10)

    # Create a mapper and actor for the plane
    plane_mapper = vtk.vtkPolyDataMapper()
    plane_mapper.SetInputConnection(plane.GetOutputPort())

    plane_actor = vtk.vtkActor()
    plane_actor.SetMapper(plane_mapper)

    # Get the points defining the plane
    origin = np.array([bounds[0], bounds[2], bounds[4] - offset])
    point1 = np.array([bounds[1], bounds[2], bounds[4] - offset])
    point2 = np.array([bounds[0], bounds[3], bounds[4] - offset])

    # Compute vectors from the origin
    vec1 = point1 - origin
    vec2 = point2 - origin

    # Compute normal
    normal = geo.compute_normal(vec1, vec2)
    plane.SetNormal(normal[0], normal[1], normal[2])

    # Get the background color from the renderer
    #background_color = renderer.GetBackground()

    # Set the plane's color to match the background
    plane_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    plane_actor.GetProperty().SetAmbient(0.7)
    plane_actor.GetProperty().SetDiffuse(0.9)
    #plane_actor.GetProperty().SetOpacity(0.5)  # 50% transparency

    plane_actor.Modified()
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


def create_mesh_and_vectorfield_actors(vertices, triangles, vertex_normals):
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

    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)

    # Create a vtkArrowSource to represent normals as arrows
    arrow_source = vtk.vtkArrowSource()
    arrow_source.SetTipResolution(16)
    arrow_source.SetShaftResolution(16)

    # Create a vtkPoints object to store the arrow positions (same as vertices)
    normal_points = vtk.vtkPoints()
    normal_vectors = vtk.vtkDoubleArray()
    normal_vectors.SetNumberOfComponents(3)  # 3D vectors

    for i in range(len(vertices)):
        normal_points.InsertNextPoint(vertices[i])
        normal_vectors.InsertNextTuple(vertex_normals[i])

    # Create polydata for the glyphs (normals visualization)
    normals_polydata = vtk.vtkPolyData()
    normals_polydata.SetPoints(normal_points)
    normals_polydata.GetPointData().SetVectors(normal_vectors)

    # Use vtkGlyph3D to generate arrows at each vertex position along the normal
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow_source.GetOutputPort())
    glyph.SetInputData(normals_polydata)
    glyph.SetVectorModeToUseVector()  # Use the normals stored in the polydata
    glyph.SetScaleFactor(0.1)  # Scale the arrows
    glyph.OrientOn()  # Align arrows with normals

    # Create a mapper and actor for the arrows (normals)
    arrow_mapper = vtk.vtkPolyDataMapper()
    arrow_mapper.SetInputConnection(glyph.GetOutputPort())

    arrow_actor = vtk.vtkActor()
    arrow_actor.SetMapper(arrow_mapper)
    arrow_actor.GetProperty().SetColor(1.0, 0.5, 0.5)  # Set arrow color to red

    return mesh_actor, arrow_actor

