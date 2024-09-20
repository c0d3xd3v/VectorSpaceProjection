import vtk

def create_bounding_box_ground_plane(renderer, offset=0.1):
    # Initialize bounds for the entire scene (actors)
    bounds = [float('inf'), -float('inf'), float('inf'), -float('inf'), float('inf'), -float('inf')]

    # Iterate over all actors to get the collective bounding box
    actors = renderer.GetActors()
    actors.InitTraversal()

    for i in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        actor_bounds = actor.GetBounds()

        # Update the global bounds
        bounds[0] = min(bounds[0], actor_bounds[0])  # X min
        bounds[1] = max(bounds[1], actor_bounds[1])  # X max
        bounds[2] = min(bounds[2], actor_bounds[2])  # Y min
        bounds[3] = max(bounds[3], actor_bounds[3])  # Y max
        bounds[4] = min(bounds[4], actor_bounds[4])  # Z min (ground)
        bounds[5] = max(bounds[5], actor_bounds[5])  # Z max

    # Create a plane at the Z-min (bottom of the bounding box) with an offset
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

    # Set the plane's color to white and make it partially transparent
    plane_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # White color
    plane_actor.GetProperty().SetOpacity(0.5)  # 50% transparency

    # Add the plane to the renderer
    renderer.AddActor(plane_actor)

def setup_lighting_and_shadow(renderer):
    # Create a light for the scene
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(10, 10, 10)
    light.SetFocalPoint(0, 0, 0)
    renderer.AddLight(light)

    # Create render passes
    shadow_pass = vtk.vtkShadowMapPass()
    opaque_pass = vtk.vtkOpaquePass()

    # Create a render pass collection
    render_pass_collection = vtk.vtkRenderPassCollection()
    render_pass_collection.AddItem(shadow_pass)
    render_pass_collection.AddItem(opaque_pass)

    seq = vtk.vtkSequencePass()
    seq.SetPasses(render_pass_collection)


    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    # Tell the renderer to use our render pass pipeline
    renderer.SetPass(cameraP)

    #render_view.SetPass(render_pass_collection)

    # Ensure the renderer is configured to use depth peeling for transparency
    renderer.SetUseDepthPeeling(False)
    renderer.SetMaximumNumberOfPeels(4)
    renderer.SetOcclusionRatio(0.1)

    # Set the background color
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background

def main():
    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add a sample actor (e.g., a sphere)
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(0.5)
    sphere_source.SetCenter(0.0, 0.0, 1.0)

    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)

    # Set sphere color
    sphere_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color

    renderer.AddActor(sphere_actor)

    # Create and add the ground plane with an offset
    create_bounding_box_ground_plane(renderer, offset=0.2)

    # Set up lighting and shadows
    setup_lighting_and_shadow(renderer)

    # Render and start interaction
    render_window.Render()
    render_window_interactor.Start()

if __name__ == '__main__':
    main()
