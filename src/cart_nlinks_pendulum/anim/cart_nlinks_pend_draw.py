from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import numpy as np
from cart_nlinks_pendulum.dynamics import CartNLinksPendPar
from dataclasses import dataclass
from typing import List
from common.geom_utils import Rect


@dataclass
class CartNLinksPendVisPar:
  lengths : List[float]
  cart_width : float  
  cart_height : float  
  wheel_radius : float
  nlinks = 0

  wheel_color : str = "#BDB4B4"
  wheel_edge_color : str = "#202020"
  cart_color : str = "#8B8BAF"
  cart_edge_color : str = '#202020'
  pend_color : str = "#4766E0"
  joints_color : str = "#C6D0F7"
  joints_edge_color : str = '#202020'

  def __post_init__(self):
    self.nlinks = len(self.lengths)
    self.lengths = np.reshape(self.lengths, (self.nlinks,))

def get_vis_par(modelpar : CartNLinksPendPar) -> CartNLinksPendVisPar:
  lengths = [l.length for l in modelpar.links]
  l = sum(lengths)
  return CartNLinksPendVisPar(
    lengths = lengths,
    cart_width = 0.30 * l,
    cart_height = 0.12 * l,
    wheel_radius = 0.04 * l,
  )

class CartNLinksPendVis:
  def __init__(self, ax : Axes, par : CartNLinksPendVisPar):
    self.__par = par
    q0 = np.zeros(par.nlinks + 1)

    coords = self.__get_verts_coords(q0)
    self.pend_line, = plt.plot(coords[:,0], coords[:,1], lw=7, color=par.pend_color)
    self.pend_joints, = plt.plot(coords[:,0], coords[:,1], 'o', 
                                 markersize=10, color=par.joints_color, markeredgecolor=par.joints_edge_color)
    p1, p2 = self.__get_wheels_coords(q0)
    self.wheel1 = Circle(p1, par.wheel_radius, color=par.wheel_color, ec=par.wheel_edge_color)
    self.wheel2 = Circle(p2, par.wheel_radius, color=par.wheel_color, ec=par.wheel_edge_color)
    ax.add_artist(self.wheel1)
    ax.add_artist(self.wheel2)

    p = self.__get_cart_pos(q0)
    self.cart = Rectangle(p, par.cart_width, par.cart_height, color=par.cart_color, ec=par.cart_edge_color)
    ax.add_artist(self.cart)

  def __get_cart_pos(self, q):
    x = q[0]
    w = self.__par.cart_width
    h = self.__par.cart_height
    return (x - w / 2, -h/2)

  def __get_wheels_coords(self, q):
    x = q[0]
    w = self.__par.cart_width
    h = self.__par.cart_height
    p1 = (x - w/3, -h*2/3)
    p2 = (x + w/3, -h*2/3)
    return p1, p2
  
  def get_covering_box(self, q : np.ndarray) -> Rect:
    p = self.__get_verts_coords(q)
    xmin,ymin = p.min(axis=0)
    xmax,ymax = p.max(axis=0)
    w = self.__par.cart_width
    h = self.__par.cart_height
    r = self.__par.wheel_radius
    xmin = min(xmin, p[0,0] - w/2)
    xmax = max(xmax, p[0,0] + w/2)
    ymin = min(ymin, p[0,1] - 2*h/3 - r)
    ymax = max(ymax, p[0,1] + h/2)
    rect = Rect(xmin, xmax, ymin, ymax)
    return rect

  def __get_verts_coords(self, q : np.ndarray) -> np.ndarray:
    nlinks = self.__par.nlinks
    nverts = nlinks + 1
    angles = np.cumsum(q[1:])
    positions = np.zeros((nverts, 2))
    positions[0,0] = q[0]
    positions[1:,0] = self.__par.lengths * np.sin(angles)
    positions[1:,1] = self.__par.lengths * np.cos(angles)
    positions = np.cumsum(positions, axis=0)
    return positions

  def move(self, q : np.ndarray):
    p1, p2 = self.__get_wheels_coords(q)
    self.wheel1.set_center(p1)
    self.wheel2.set_center(p2)
    coords = self.__get_verts_coords(q)
    self.pend_line.set_data(coords[:,0], coords[:,1])
    self.pend_joints.set_data(coords[:,0], coords[:,1])
    p = self.__get_cart_pos(q)
    self.cart.set_xy(p)

  @property
  def patches(self):
    return self.wheel1, self.wheel2, self.cart, self.pend_line, self.pend_joints
