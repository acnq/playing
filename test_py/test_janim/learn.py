from janim.imports import *

class HelloJAnimExample(Timeline):
  def construct(self):
    # define items
    circle = Circle(color=BLUE)
    square = Square(color=GREEN, fill_alpha=0.5)
    
    # do animations
    self.forward()
    self.play(Create(circle))
    self.play(Transform(circle, square))
    self.play(Uncreate(square))
    self.forward()
    
class CmptAnimExample(Timeline):
  def construct(self) -> None:
    circle = Circle(color=BLUE, fill_alpha=0.5)
    
    self.show(circle)
    self.forward()
    self.play(circle.anim.color.set(GREEN))
    self.play(circle.anim.fill.set(alpha=0.2))
    self.play(circle.anim.points.scale(2))
    self.forward()