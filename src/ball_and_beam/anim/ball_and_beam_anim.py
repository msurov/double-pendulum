from matplotlib.axes import Axes
from common.trajectory import Trajectory
from ball_and_beam.anim import BallAndBeamVis, BallAndBeamVisPar, get_view_box
from scipy.interpolate import make_interp_spline
from visualization.anim import Animate, AnimatorObject
import matplotlib.pyplot as plt
from common.geom_utils import Rect, covering_rect, inflate_rect
from functools import reduce


class BallAndBeamAnim(AnimatorObject):
  def __init__(self, ax : Axes, par : BallAndBeamVisPar, traj : Trajectory):
    self.__model = BallAndBeamVis(ax, par)
    self.__sp = make_interp_spline(traj.time, traj.coords, k=1)
    self.__sp.extrapolate = None

  def update(self, t):
    q = self.__sp(t)
    self.__model.move(q)

  @property
  def patches(self):
    return self.__model.patches

def get_traj_view_box(traj : Trajectory, vispar : BallAndBeamVisPar) -> Rect:
  r0 = get_view_box(traj.coords[0], vispar)
  return reduce(lambda rect, q: covering_rect(get_view_box(q, vispar), rect), traj.coords, r0)

def launch_anim(traj : Trajectory, par : BallAndBeamVisPar, **kwargs):
  fig = plt.gcf()
  ax = plt.gca()
  ax.set_aspect(1)
  box = get_traj_view_box(traj, par)
  box = inflate_rect(box, 1.1)
  ax.set_xlim(box.x1, box.x2)
  ax.set_ylim(box.y1, box.y2)
  cart_pend_anim = BallAndBeamAnim(ax, par, traj)
  return Animate(fig, [cart_pend_anim], traj.time[-1], **kwargs)
