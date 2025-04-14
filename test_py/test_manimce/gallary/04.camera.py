from manim import *

class FollowingGraphCamera(MovingCameraScene):
  def construct(self):
    self.camera.frame.save_state()
    
    # create the axes and the curve
    ax = Axes(x_range=[-1, 10], y_range=[-1, 10])
    graph = ax.plot(lambda x: np.sin(x), color=BLUE, x_range=[0, 3 * PI])
    
    # create dots based on the graph
    moving_dot = Dot(ax.i2gp(graph.t_min, graph), color=ORANGE)
    dot_1 = Dot(ax.i2gp(graph.t_min, graph))
    dot_2 = Dot(ax.i2gp(graph.t_max, graph))
    
    self.add(ax, graph, dot_1, dot_2, moving_dot)
    self.play(self.camera.frame.animate.scale(0.5).move_to(moving_dot))
    
    def update_curve(mob):
      mob.move_to(moving_dot.get_center())
      
    self.camera.frame.add_updater(update_curve)
    self.play(MoveAlongPath(moving_dot, graph, rate_func=linear))
    self.camera.frame.remove_updater(update_curve)
    
    self.play(Restore(self.camera.frame))
    
class MovingZoomedSceneAround(ZoomedScene):
  # contributed by TheoremofBeethoven, www.youtube.com/c/TheoremofBeethoven
  def __init__(self, **kwargs):
    ZoomedScene.__init__(
      self,
      zoom_factor=0.3,
      zoomed_display_height=1,
      zoomed_display_width=6,
      image_frame_stroke_width=20,
      zoomed_camera_config={
        "default_frame_stroke_width": 3,
      },
      **kwargs
    )
    
  def construct(self):
    dot = Dot().shift(UL * 2)
    image = ImageMobject(np.uint8([ [0, 100, 30, 200],
                                    [255, 0, 5, 33]]))
    image.height = 7
    frame_text = Text("Frame", color=PURPLE, font_size=67)
    zoomed_camera_text = Text("Zoomed camera", color=RED, font_size=67)
    
    self.add(image, dot)
    zoomed_camera = self.zoomed_camera
    zoomed_display = self.zoomed_display
    frame = zoomed_camera.frame
    zoomed_display_frame = zoomed_display.display_frame
    
    frame.move_to(dot)
    frame.set_color(PURPLE)
    zoomed_display_frame.set_color(RED)
    zoomed_display.shift(DOWN)
    
    zd_rect = BackgroundRectangle(zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF)
    self.add_foreground_mobject(zd_rect)
    
    unfold_camera = UpdateFromFunc(zd_rect, lambda rect: rect.replace(zoomed_display))
    
    frame_text.next_to(frame, DOWN)
    
    self.play(Create(frame), FadeIn(frame_text, shift=UP))
    self.activate_zooming()
    
    self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera)
    zoomed_camera_text.next_to(zoomed_display_frame, DOWN)
    self.play(FadeIn(zoomed_camera_text, shift=UP))
    # Scale in        x    y  z
    scale_factor = [0.5, 1.5, 0]
    self.play(
      frame.animate.scale(scale_factor),
      zoomed_display.animate.scale(scale_factor),
      FadeOut(zoomed_camera_text),
      FadeOut(frame_text)
    )
    self.wait()
    self.play(ScaleInPlace(zoomed_display, 2))
    self.wait()
    self.play(frame.animate.shift(2.5 * DOWN))
    self.wait()
    self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera, rate_func=lambda t: smooth(1 - t))
    self.play(Uncreate(zoomed_display_frame), FadeOut(frame))
    self.wait()
    
class FixedInFrameMObjectTests(ThreeDScene):
  def construct(self):
    axes = ThreeDAxes()
    self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
    text3d = Text("This is a 3D text")
    self.add_fixed_in_frame_mobjects(text3d)
    text3d.to_corner(UL)
    self.add(axes)
    self.wait()
    
class ThreeDLightSourcePosition(ThreeDScene):
  def construct(self):
    axes = ThreeDAxes()
    sphere = Surface(
      lambda u, v: np.array([
        1.5 * np.cos(u) * np.cos(v),
        1.5 * np.cos(u) * np.sin(v),
        1.5 * np.sin(u)
      ]), v_range=[0, TAU], u_range=[-PI / 2, PI / 2],
      checkerboard_colors=[RED_D, RED_E], resolution=(15, 32)
    )
    self.renderer.camera.light_source.move_to(3 * IN) 
    # changes the source of the light
    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
    self.add(axes, sphere)
    
class ThreeDCameraRotation(ThreeDScene):
  def construct(self):
    axes = ThreeDAxes()
    circle = Circle()
    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
    self.add(circle, axes)
    self.begin_ambient_camera_rotation(rate=0.1)
    self.wait()
    self.stop_ambient_camera_rotation()
    self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
    self.wait()
    
class ThreeDCameraIllusionRotation(ThreeDScene):
  def construct(self):
    axes = ThreeDAxes()
    circe = Circle()
    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
    self.add(circe, axes)
    self.begin_3dillusion_camera_rotation(rate=2)
    self.wait(PI/2)
    self.stop_3dillusion_camera_rotation()
    
class ThreeDSurfacePlot(ThreeDScene):
  def construct(self):
    resolution_fa = 24
    self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
    
    def param_gauss(u, v):
      x = u
      y = v
      sigma, mu = 0.4, [0.0, 0.0]
      d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
      z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
      return np.array([x, y, z])
    
    gauss_plane = Surface(
      param_gauss,
      resolution=(resolution_fa, resolution_fa),
      v_range=[-2, +2],
      u_range=[-2, +2]
    )
    
    gauss_plane.scale(2, about_point=ORIGIN)
    gauss_plane.set_style(fill_opacity=1, stroke_color=GREEN)
    gauss_plane.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
    axes = ThreeDAxes()
    self.add(axes, gauss_plane)