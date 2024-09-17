import numpy as np
import vtk

def create_light_source_at_camera(renderer, camera):
    """Creates a light source and positions it at the camera's position."""
    # Create a light source
    colors = vtk.vtkNamedColors()
    colors.SetColor('HighNoonSun', [255, 255, 251, 255])  # Color temp. 5400°K
    colors.SetColor('100W Tungsten', [255, 214, 170, 255])  # Color temp. 2850°K

    light2 = vtk.vtkLight()
    light2.SetFocalPoint(0, 0, 0)
    light2.SetPosition(2.0, -3.0, -4.0)
    light2.SetColor(colors.GetColor3d('100W Tungsten'))
    light2.SetIntensity(0.25)
    renderer.AddLight(light2)

    # Function to update light position
    def update_light_position(*args):
        #print("update")
        light_position = camera.GetPosition()
        light_position += np.array([3.5, 1.5, 1.0])
        light2.SetPosition(light_position)
        renderer.GetRenderWindow().Render()  # Re-render to apply the changes

    # Set the initial position
    #update_light_position()
    camera.AddObserver('ModifiedEvent', update_light_position)

    return light2, update_light_position


def render_actors(actors, camera_position=[1, 0, 0]):
    # VTK Visualization
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1,1,1)  # Hintergrundfarbe auf weiß setzen

    cam = renderer.GetActiveCamera()
    cam.SetViewUp(camera_position)

    # Renderer hinzufügen
    cube_axes_actor = createCubeAxesActor(renderer)
    #renderer.AddActor(cube_axes_actor)

    for actor in actors:
        actor.GetProperty().SetAmbient(0.7)
        actor.GetProperty().SetDiffuse(0.9)
        renderer.AddActor(actor)

    # Renderer-Instanz erstellen
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    #renderer.UseFXAAOn()

    # RenderWindowInteractor erstellen
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Konfiguriere Depth Testing
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(100.5)

    light, update_light_position = create_light_source_at_camera(renderer, renderer.GetActiveCamera())

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()

    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)

    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    # Tell the renderer to use our render pass pipeline
    renderer.SetPass(cameraP)

    render_window.SetMultiSamples(8)
    # Starten Sie das Rendering
    render_window.Render()
    render_window_interactor.Start()


def createCubeAxesActor(renderer):
    # CubeAxesActor erstellen
    cube_axes_actor = vtk.vtkCubeAxesActor()
    cube_axes_actor.SetCamera(renderer.GetActiveCamera())
    cube_axes_actor.SetXTitle("X-Axis")
    cube_axes_actor.SetYTitle("Y-Axis")
    cube_axes_actor.SetZTitle("Z-Axis")
    cube_axes_actor.SetFlyModeToOuterEdges()
    # Achsen- und Schriftfarbe auf schwarz setzen
    cube_axes_actor.SetDrawXGridlines(True)
    cube_axes_actor.SetDrawYGridlines(True)
    cube_axes_actor.SetDrawZGridlines(True)

    b = 0.5
    cube_axes_actor.SetBounds(-b, b, -b, b, -b, b)

    cube_axes_actor.GetXAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
    cube_axes_actor.GetYAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
    cube_axes_actor.GetZAxesLinesProperty().SetColor(0.0, 0.0, 0.0)

    cube_axes_actor.GetXAxesGridlinesProperty().SetColor(0, 0,0)
    cube_axes_actor.GetYAxesGridlinesProperty().SetColor(0, 0,0)
    cube_axes_actor.GetZAxesGridlinesProperty().SetColor(0, 0,0)

    cube_axes_actor.GetTitleTextProperty(0).SetColor(0.0, 0.0, 0.0)
    cube_axes_actor.GetTitleTextProperty(1).SetColor(0.0, 0.0, 0.0)
    cube_axes_actor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 0.0)

    cube_axes_actor.GetLabelTextProperty(0).SetColor(0.0, 0.0, 0.0)
    cube_axes_actor.GetLabelTextProperty(1).SetColor(0.0, 0.0, 0.0)
    cube_axes_actor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 0.0)

    cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)


    return cube_axes_actor


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


def add_plane(renderer, n, color_plane):
    # Erstellen Sie die Linien für die Vektoren
    c2 = color_plane
    c2 = [
    (c2[0] + 1) * 0.5,
    (c2[1] + 1) * 0.5,
    (c2[2] + 1) * 0.5]

    # Erstellen Sie die Ebene
    actor_plane = create_plane_actor(n, [0.0, 0.0, 0.0], color_plane)
    renderer.AddActor(actor_plane)

