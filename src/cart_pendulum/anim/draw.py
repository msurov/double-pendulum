from matplotlib.patches import Circle, Rectangle
from dataclasses import dataclass

@dataclass
class CartPendulumVisPar:
  cart_width : float
  cart_height : float
  pendulum_length : float
  wheel_radius : float

class CartPendulumVis:
  def __init__(self, ax, par : CartPendulumVisPar):
    self.par = par

  def get_wheels_coords(self, x):
    w = -self.par.wheel_radius - self.par.cart_height
    p1 = np.array([
      x - w, -h
    ])
    p1 = np.array([
      x + w, -h
    ])

  def create_body(self):
    p = self.get_body_coords(0)
    body = Rectangle(p, self.par.cart_width, self.par.cart_height)
    return body

  def get_body_coords(self, x):
    width = self.par.cart_width
    height = self.par.cart_height
    p = np.array([
      x - width/2, 0 - height/2
    ])
    return p
  
  def create_rod(self):
    length = self.par.pendulum_length
    p, angle = self.get_rod_coords(0, 0)
    rod = Rectangle(p, 5e-2 * length, length, angle)
    return rod

  def get_rod_coords(self, x, theta):
    p = np.array([x, 0])
    return p, theta

  def move(self, q):
    pass
