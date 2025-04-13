from manim import *

class SinAndCosFunctionPlot(Scene):
  def construct(self):
    axes = Axes(
      x_range=[-10, 10.3, 1],
      y_range=[-1.5, 1.5, 1],
      x_length=10,
      axis_config={"color": GREEN},
      x_axis_config={
        "numbers_to_include": np.arange(-10, 10.01, 2),
        "numbers_with_elongated_ticks": np.arange(-10, 10.01, 2),
      },
      tips=False,
    )
    axes_labels = axes.get_axis_labels()
    sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
    cos_graph = axes.plot(lambda x: np.cos(x), color=RED)
    
    sin_label = axes.get_graph_label(
      sin_graph, "\\sin(x)", x_val=-10, direction=UP / 2
    )
    cos_label = axes.get_graph_label(cos_graph, label="\\cos(x)")
    vert_line = axes.get_vertical_line(
      axes.i2gp(TAU, cos_graph), color=YELLOW, line_func=Line
    )
    line_label = axes.get_graph_label(
      cos_graph, r"x=2\pi", x_val=TAU, direction=UR, color=WHITE
    )
    plot = VGroup(axes, sin_graph, cos_graph, vert_line)
    labels = VGroup(axes_labels, sin_label, cos_label, line_label)
    self.add(plot, labels)
    
class ArgMinExample(Scene):
  def construct(self):
    ax = Axes(
      x_range=[0, 10], y_range=[0, 100, 10], axis_config={"include_tip": False}
    )
    labels = ax.get_axis_labels(x_label="x", y_label="f(x)")
    t = ValueTracker(0)
    
    def func(x):
      return 2 * (x - 5) ** 2
    graph = ax.plot(func, color=MAROON)
    initial_point = [ax.coords_to_point(t.get_value(), func(t.get_value()))]
    dot = Dot(point=initial_point)
    
    dot.add_updater(lambda x: x.move_to(ax.c2p(t.get_value(), func(t.get_value()))))
    x_space = np.linspace(*ax.x_range[:2], 200)
    minimum_index = func(x_space).argmin()
    
    self.add(ax, labels, graph, dot)
    self.play(t.animate.set_value(x_space[minimum_index]))
    self.wait()               
    
class GraphAreaPlot(Scene):
  def construct(self):
    ax = Axes(
      x_range=[0, 5],
      y_range=[0, 6],
      x_axis_config={"numbers_to_include": [2, 3]},
      tips=False,
    )
    labels = ax.get_axis_labels()
    curve_1 = ax.plot(lambda x: 4 * x - x ** 2, x_range=[0, 4], color=BLUE_C)
    curve_2 = ax.plot(
      lambda x: 0.8 * x ** 2 - 3 * x  + 4,
      x_range=[0, 4],
      color=GREEN_B,
    )
    line_1 = ax.get_vertical_line(ax.input_to_graph_point(2, curve_1), color=YELLOW)
    line_2 = ax.get_vertical_line(ax.i2gp(3, curve_1), color=YELLOW)
    
    riemann_area = ax.get_riemann_rectangles(curve_1, x_range=[0.3, 0.6], dx=0.03, color=BLUE, fill_opacity=0.5)
    area = ax.get_area(curve_2, [2, 3], bounded_graph=curve_1, color=GREY, opacity=0.5)
    
    self.add(ax, labels, curve_1, curve_2, line_1, line_2, riemann_area, area)
    
class PolygonOnAxes(Scene):
  def get_rectangle_corners(self, bottom_left, top_right):
    return [
      (top_right[0], top_right[1]),
      (bottom_left[0], top_right[1]),
      (bottom_left[0], bottom_left[1]),
      (top_right[0], bottom_left[1]),
    ]
    
  def construct(self):
    ax = Axes(
      x_range=[0, 10],
      y_range=[0, 10],
      x_length=6, 
      y_length=6,
      axis_config={"include_tip": False},
    )
    
    t = ValueTracker(5)
    k = 25
    
    graph = ax.plot(
      lambda x: k / x,
      color=YELLOW_D,
      x_range=[k / 10, 10.0, 0.01],
      use_smoothing=False,
    )
    
    def get_rectangle():
      polygon = Polygon(
        *[
          ax.c2p(*i)
          for i in self.get_rectangle_corners(
            (0, 0), (t.get_value(), k / t.get_value())
          )
        ]
      )
      polygon.stroke_width = 1
      polygon.set_fill(BLUE, opacity=0.5)
      polygon.set_stroke(YELLOW_B)
      return polygon
    
    polygon = always_redraw(get_rectangle)
    
    dot = Dot()
    dot.add_updater(lambda x: x.move_to(ax.c2p(t.get_value(), k / t.get_value())))
    dot.set_z_index(10)
    
    self.add(ax, graph, dot)
    self.play(Create(polygon))
    self.play(t.animate.set_value(10))
    self.play(t.animate.set_value(k / 10))
    self.play(t.animate.set_value(5))
    

class HeatDiagramPlot(Scene):
  def construct(self):
    ax = Axes(
      x_range=[0, 40, 5],
      y_range=[-8, 32, 5],
      x_length=9,
      y_length=6,
      x_axis_config={"numbers_to_include": np.arange(0, 40, 5)},
      y_axis_config={"numbers_to_include": np.arange(-5, 34, 5)},
      tips=False,
    )
    labels = ax.get_axis_labels(
      x_label=Tex(r"$\Delta Q$"), y_label=Tex(r"T[$^\circ C$]")
    )
    x_vals = [0, 8, 38, 39]
    y_vals = [20, 0, 0, -5]
    graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals)
    
    self.add(ax, labels, graph)