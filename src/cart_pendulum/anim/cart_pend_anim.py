from matplotlib.axes import Axes
from common.trajectory import Trajectory
from cart_pendulum.anim import CartPendulumVis, CartPendulumVisPar, get_view_box
from scipy.interpolate import make_interp_spline
from visualization.anim import Animate, AnimatorObject
import matplotlib.pyplot as plt
from common.geom_utils import Rect, covering_rect


class CartPendulumAnim(AnimatorObject):
  def __init__(self, ax : Axes, par : CartPendulumVisPar, traj : Trajectory):
    self.__model = CartPendulumVis(ax, par)
    self.__sp = make_interp_spline(traj.time, traj.coords, k=1)
    self.__sp.extrapolate = None

  def update(self, t):
    q = self.__sp(t)
    self.__model.move(q)

  @property
  def patches(self):
    return self.__model.patches

def get_traj_view_box(traj : Trajectory, par : CartPendulumVisPar) -> Rect:
  box = get_view_box(traj.coords[0], par)
  for q in traj.coords:
    box = covering_rect(box, get_view_box(q, par))
  return box

def launch_anim(traj : Trajectory, par : CartPendulumVisPar):
  fig, ax = plt.subplots(1, 1)
  ax.set_aspect(1)
  box = get_traj_view_box(traj, par)
  ax.set_xlim(box.x1, box.x2)
  ax.set_ylim(box.y1, box.y2)
  cart_pend_anim = CartPendulumAnim(ax, par, traj)
  return Animate(fig, [cart_pend_anim], traj.time[-1])
