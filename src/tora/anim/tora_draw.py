from matplotlib.patches import Circle, Rectangle
from matplotlib.axes import Axes
import numpy as np
from dataclasses import dataclass
from tora.dynamics import TORAPar
from typing import Tuple, List
from common.geom_utils import Rect

sin = np.sin
cos = np.cos
rad2deg = np.rad2deg

@dataclass
class ToraVisPar:
  cart_width : float
  cart_height : float
  rod_length : float | List[float]
  rod_width : float
  wheel_radius : float

  wheel_color = '#404040'
  body_color = '#4040A0'
  rod_color = '#40A0A0'
  pend_color = "#2B7A7A"
  alpha = 1

def get_vis_par(systempar : TORAPar):
  l = systempar.pendulum_length
  return ToraVisPar(
    cart_width = 0.4 * l,
    cart_height = 0.16 * l,
    rod_length = l,
    rod_width = 3e-2 * l,
    wheel_radius = 5e-2 * l,
  )

def get_view_box(q : np.ndarray, par : ToraVisPar) -> Rect:
  """
    returns xmin, xmax, ymin, ymax
  """
  l = par.rod_length
  w = par.cart_width
  h = par.cart_height
  r = par.wheel_radius
  x = [
    q[0] + w / 2, q[0] - w / 2, q[0] - l * sin(q[1]) + par.rod_width, q[0] - l * sin(q[1]) - par.rod_width
  ]
  y = [
    -h/2 - r, h/2 + r, -l * cos(q[1]) + par.rod_width, -l * cos(q[1]) - par.rod_width
  ]
  xmin = np.min(x)
  xmax = np.max(x)
  ymin = np.min(y)
  ymax = np.max(y)
  return Rect(xmin, xmax, ymin, ymax)

class ToraVis:
  def __init__(self, ax : Axes, par : ToraVisPar):
    self.__par = par
    self.__create_body()
    self.__create_wheels()
    self.__create_rod()
    for e in self.patches:
      ax.add_artist(e)
  
  @property
  def patches(self):
    return self.__wheel1, self.__wheel2, self.__body, self.__rod, self.__pend_c1, self.__pend_c2

  def __get_wheels_coords(self, x):
    w = self.__par.cart_width / 3
    p1 = np.array([
      x - w, -self.__par.cart_height / 2
    ])
    p2 = np.array([
      x + w, -self.__par.cart_height / 2
    ])
    return p1, p2
  
  def __create_wheels(self):
    p1, p2 = self.__get_wheels_coords(0)
    self.__wheel1 = Circle(p1, self.__par.wheel_radius, color=self.__par.wheel_color, alpha=self.__par.alpha)
    self.__wheel2 = Circle(p2, self.__par.wheel_radius, color=self.__par.wheel_color, alpha=self.__par.alpha)

  def __get_body_coords(self, x):
    w = self.__par.cart_width / 2
    h = self.__par.cart_height / 2
    p = np.array([
      x - w, -h
    ])
    return p

  def __create_body(self):
    p = self.__get_body_coords(0)
    self.__body = Rectangle(xy = p, width = self.__par.cart_width, height = self.__par.cart_height, angle = 0., color=self.__par.body_color, alpha=self.__par.alpha)

  def __get_rod_coords(self, x, theta):
    w = 0.5 * self.__par.rod_width
    p = np.array([x + w * cos(theta), w * sin(theta)])
    return p, 180 + rad2deg(theta)
  
  def __get_pend_coords(self, x, theta):
    p1 = np.array([x, 0])
    p2 = p1 + self.__par.rod_length * np.array([sin(theta), -cos(theta)])
    return p1, p2

  def __create_rod(self):
    length = self.__par.rod_length
    p, angle = self.__get_rod_coords(0, 0)
    self.__rod = Rectangle(xy = p, width = self.__par.rod_width, height = length, angle = angle, color=self.__par.rod_color, alpha=self.__par.alpha)
    p1,p2 = self.__get_pend_coords(0, 0)
    self.__pend_c1 = Circle(xy = p1, radius = self.__par.rod_width, color=self.__par.pend_color, alpha=self.__par.alpha)
    self.__pend_c2 = Circle(xy = p2, radius = self.__par.rod_width, color=self.__par.pend_color, alpha=self.__par.alpha)

  def move(self, q):
    x, theta = q
    p1, p2 = self.__get_wheels_coords(x)
    self.__wheel1.set_center(p1)
    self.__wheel2.set_center(p2)
    self.__body.set_xy(self.__get_body_coords(x))
    p,a = self.__get_rod_coords(x, theta) 
    self.__rod.set_xy(p)
    self.__rod.set_angle(a)
    p1,p2 = self.__get_pend_coords(x, theta)
    self.__pend_c1.set_center(p1)
    self.__pend_c2.set_center(p2)
