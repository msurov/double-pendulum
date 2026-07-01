from matplotlib.patches import Circle, Rectangle, Arc, Wedge, Polygon
from matplotlib.transforms import Affine2D
from matplotlib.collections import PatchCollection
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from ball_and_beam.dynamics import BallAndBeamPar
from typing import Tuple, List
from common.geom_utils import Rect
from common.numpy_utils import rotmat2d
from functools import reduce
from common.geom_utils import Rect, covering_rect

sin = np.sin
cos = np.cos
rad2deg = np.rad2deg

@dataclass
class BallAndBeamVisPar:
  ball_radius : float
  beam_thickness : float
  beam_length : float
  surface_vertical_displacement : float
  joint_radius : float

  ball_color = '#404040'
  beam_color = "#7D7DFB"
  joint_color = '#202020'

def get_vis_par(systempar : BallAndBeamPar):
  return BallAndBeamVisPar(
    ball_radius = systempar.ball_radius,
    beam_thickness = 0.5 * systempar.ball_radius,
    beam_length = 10 * systempar.ball_radius,
    joint_radius = 0.1 * systempar.ball_radius,
    surface_vertical_displacement = systempar.ball_center_displacement - systempar.ball_radius
  )

def get_view_box(q : np.ndarray, par : BallAndBeamVisPar) -> Rect:
  """
    returns xmin, xmax, ymin, ymax
  """
  theta, s = q
  l = par.beam_length
  w = par.beam_thickness
  r = par.ball_radius

  R = rotmat2d(theta)
  p_ball = R @ np.array([s, par.surface_vertical_displacement + par.ball_radius])
  p1 = R @ np.array([par.beam_length / 2, par.surface_vertical_displacement])
  p2 = R @ np.array([-par.beam_length / 2, par.surface_vertical_displacement])
  p3 = R @ np.array([par.beam_length / 2, par.surface_vertical_displacement - par.beam_thickness])
  p4 = R @ np.array([-par.beam_length / 2, par.surface_vertical_displacement - par.beam_thickness])

  x = [
    0,
    p_ball[0] + par.ball_radius,
    p_ball[0] - par.ball_radius,
    p1[0], p2[0], p3[0], p4[0]
  ]

  y = [
    -par.beam_thickness,
    p_ball[1] + par.ball_radius,
    p_ball[1] - par.ball_radius,
    p1[1], p2[1], p3[1], p4[1]
  ]

  xmin = np.min(x)
  xmax = np.max(x)
  ymin = np.min(y)
  ymax = np.max(y)
  return Rect(xmin, xmax, ymin, ymax)

class BallAndBeamVis:
  def __init__(self, ax : Axes, par : BallAndBeamVisPar):
    self.__par = par
    self.__ax = ax
    self.__create_beam()
    self.__create_ball()
    self.__create_joint()

    for p in self.patches:
      ax.add_artist(p)

  @property
  def patches(self):
    return self.__ball, self.__beam, self.__joint
  
  def __create_joint(self):
    self.__joint = Circle([0, 0], self.__par.joint_radius, color=self.__par.joint_color)

  def __get_beam_coords(self, q):
    theta, _ = q
    R = rotmat2d(theta)
    corner = np.array([
      -self.__par.beam_length / 2,
      self.__par.surface_vertical_displacement - self.__par.beam_thickness
    ])
    rect_lower_left = R @ corner
    return rect_lower_left, theta
  
  def __create_beam(self):
    assert self.__par.surface_vertical_displacement >= 0
    pts = np.array([
      [-self.__par.beam_length / 2, self.__par.surface_vertical_displacement],
      [self.__par.beam_length / 2, self.__par.surface_vertical_displacement],
      [self.__par.beam_length / 2, self.__par.surface_vertical_displacement - self.__par.beam_thickness],
      [self.__par.beam_thickness, -self.__par.beam_thickness],
      [-self.__par.beam_thickness, -self.__par.beam_thickness],
      [-self.__par.beam_length / 2, self.__par.surface_vertical_displacement - self.__par.beam_thickness],
    ])
    self.__beam = Polygon(pts, closed=True, fc=self.__par.beam_color)

  def __get_ball_coords(self, q):
    theta, s = q
    ball_center_proj = self.__par.surface_vertical_displacement + self.__par.ball_radius
    p = rotmat2d(theta) @ np.array([s, ball_center_proj])
    angle = -s / self.__par.ball_radius + theta
    return p, angle

  def __create_ball(self):
    p, a = self.__get_ball_coords([0, 0])
    nsectors = 6
    colors = ["#FF7373", "#009DB1"]

    wedges = [Wedge([0, 0], self.__par.ball_radius, i * 360 / nsectors, (i + 1) * 360 / nsectors) 
     for i in range(nsectors)]
  
    self.__ball = PatchCollection(wedges, facecolors=colors)
    T = Affine2D(np.array([
      [cos(a), -sin(a), p[0]],
      [sin(a), cos(a), p[1]],
      [0, 0, 1]
    ]))
    self.__ball.set_transform(T + self.__ax.transData)

  def move(self, q):
    theta, s = q

    p, a = self.__get_beam_coords(q)
    # self.__beam.set_xy(p)
    # self.__beam.set_angle(rad2deg(a))
    T = Affine2D()
    T.rotate(theta)
    self.__beam.set_transform(T + self.__ax.transData)

    p, a = self.__get_ball_coords(q)
    T = Affine2D(np.array([
      [cos(a), -sin(a), p[0]],
      [sin(a), cos(a), p[1]],
      [0, 0, 1]
    ]))
    self.__ball.set_transform(T + self.__ax.transData)
