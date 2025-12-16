from functools import reduce
from matplotlib.axes import Axes
from common.trajectory import Trajectory
from .cart_nlinks_pend_draw import (
  CartNLinksPendPar,
  CartNLinksPendVis,
  CartNLinksPendVisPar,
  get_vis_par
)
from scipy.interpolate import make_interp_spline
from visualization.anim import Animate, AnimatorObject
import matplotlib.pyplot as plt
from common.geom_utils import Rect, covering_rect


class CartNLinksPendAnim(AnimatorObject):
  def __init__(self, ax : Axes, par : CartNLinksPendVisPar, traj : Trajectory):
    self.__model = CartNLinksPendVis(ax, par)
    self.__sp = make_interp_spline(traj.time, traj.coords, k=1)
    self.__sp.extrapolate = None
    self.fit_view(ax, traj)

  def get_covering_box(self, traj : Trajectory) -> Rect:
    return reduce(
      lambda box, q: covering_rect(box, self.__model.get_covering_box(q)), 
      traj.coords[1:,:],
      self.__model.get_covering_box(traj.coords[0])
    )

  def fit_view(self, ax : Axes, traj : Trajectory):
    rect = self.get_covering_box(traj)
    ax.set_xlim(rect.x1, rect.x2)
    ax.set_ylim(rect.y1, rect.y2)

  def update(self, t):
    q = self.__sp(t)
    self.__model.move(q)

  @property
  def patches(self):
    return self.__model.patches

def launch_anim(traj : Trajectory, modelpar : CartNLinksPendPar) -> Animate:
  fig, ax = plt.subplots(1, 1)
  ax.set_aspect(1)
  vispar = get_vis_par(modelpar)
  cart_pend_anim = CartNLinksPendAnim(ax, vispar, traj)
  return Animate(fig, [cart_pend_anim], traj.time[-1])
