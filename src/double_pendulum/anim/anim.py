import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from scipy.interpolate import make_interp_spline
from double_pendulum.dynamics import DoublePendulumParam, double_pendulum_param_default
from common.trajectory import Trajectory
from matplotlib.patches import Rectangle, Circle
from visualization import Animate, AnimGraph
from numbers import Number
from typing import Union, Tuple
from dataclasses import dataclass

class Link:
  def __init__(self, width : float, length : float, **patch_params):
    origin = np.zeros(2)

    self.angle = 0
    self.origin = origin

    d = np.array([width/2, 0])
    p = origin - d
    rect = Rectangle(p, width, length, rotation_point='xy', **patch_params)

    self.rect = rect
    self.width = width
    self.length = length

  def move(self, origin : Tuple[float,float], angle : float):
    self.origin = origin
    p = self.origin + self.width/2 * np.array([-np.cos(angle), np.sin(angle)])
    self.rect.set_xy(p)
    self.rect.set_angle(-np.rad2deg(angle))

  @property
  def patches(self):
    return self.rect,

class Joint:
  def __init__(self, diameter : float, **patch_params):
    origin = np.zeros(2)
    self.origin = origin
    self.radius = diameter/2
    self.circle = Circle(self.origin, self.radius, **patch_params)
  
  def move(self, center : Tuple[float,float]):
    self.origin = center
    self.circle.set_center(self.origin)

  @property
  def patches(self):
    return self.circle,

@dataclass
class DoublePendulumViewParam:
  links_face : Tuple[Union[float,str]] = ('#A0A0FF',) * 2
  links_edge : Tuple[Union[float,str]] = ('#202020',) * 2
  joints_face : Tuple[Union[float,str]] = ('#808080',) * 3
  joints_edge : Tuple[Union[float,str]] = ('#202020',) * 3
  links_length : Tuple[Number] = (1,) * 2
  links_width : Tuple[Number] = (0.1,) * 2
  joints_radius : Tuple[Number] = (0.1,) * 3
  alpha : float = 1

def compute_viewbox(q : np.ndarray, pr : DoublePendulumViewParam):
  theta1 = q[...,0]
  theta2 = q[...,1]
  l1, l2 = pr.links_length

  x1 = l1 * np.sin(theta1)
  y1 = l1 * np.cos(theta1)
  x2 = x1 + l2 * np.sin(theta1 + theta2)
  y2 = y1 + l2 * np.cos(theta1 + theta2)
  xmin = min(np.min(x1), np.min(x2), 0)
  xmax = max(np.max(x1), np.max(x2), 0)
  ymin = min(np.min(y1), np.min(y2), 0)
  ymax = max(np.max(y1), np.max(y2), 0)
  return xmin, xmax, ymin, ymax

def inflate_viewbox(xmin, xmax, ymin, ymax, pcnt):
  w = xmax - xmin
  h = ymax - ymin
  cx = (xmax + xmin) / 2
  cy = (ymax + ymin) / 2
  w1 = w * (1 + pcnt / 100)
  h1 = h * (1 + pcnt / 100)
  return cx - w1/2, cx + w1/2, cy - h1/2, cy + h1/2

def get_view_parameters(par : DoublePendulumParam) -> DoublePendulumViewParam:
  l1, l2 = par.lengths
  w1 = l1/10
  w2 = l2/10
  return DoublePendulumViewParam(
    links_length = [l1, l2],
    links_width = [w1, w2],
    joints_radius = [1.2 * w1, 1.2 * w2, 1.2 * w2]
  )

class DoublePendulumView:
  def __init__(self, view_par : DoublePendulumViewParam):
    self.l1, self.l2 = view_par.links_length
    self.w1, self.w2 = view_par.links_width
    r1, r2, r3 = view_par.joints_radius
    alpha = view_par.alpha

    self.objects = [
      Link(self.w1, self.l1, fc=view_par.links_face[0], ec=view_par.links_edge[0], alpha=alpha),
      Link(self.w2, self.l2, fc=view_par.links_face[1], ec=view_par.links_edge[1], alpha=alpha),
      Joint(r1, fc=view_par.joints_face[0], ec=view_par.joints_edge[0]),
      Joint(r2, fc=view_par.joints_face[1], ec=view_par.joints_edge[1]),
      Joint(r3, fc=view_par.joints_face[2], ec=view_par.joints_edge[2]),
    ]
    self.move([0, 0])

  def get_objects_poses(self, q):
    x1 = self.l1 * np.sin(q[0])
    y1 = self.l1 * np.cos(q[0])
    x2 = x1 + self.l2 * np.sin(q[0] + q[1])
    y2 = y1 + self.l2 * np.cos(q[0] + q[1])

    link1_pos = ([0, 0], q[0])
    link2_pos = ([x1, y1], q[0] + q[1])
    joint1_pos = ([0, 0],)
    joint2_pos = ([x1, y1],)
    joint3_pos = ([x2, y2],)

    return link1_pos, link2_pos, joint1_pos, joint2_pos, joint3_pos

  def move(self, q):
    poses = self.get_objects_poses(q)
    for obj, pos in zip(self.objects, poses):
      obj.move(*pos)

  @property
  def patches(self):
    patches = reduce(lambda acc, obj: acc + obj.patches, self.objects, tuple())
    return tuple(patches)

class DoublePendulumAnim:
  def __init__(self, ax, view_par : DoublePendulumViewParam, traj : Trajectory, 
                shadow_color = '#F0F0F0', 
                vertsize = 12, linkwidth = 4, link_color = 'darkblue'):

    self.traj_sp = make_interp_spline(traj.time, traj.coords, k=1)
    self.traj_sp.extrapolate = None
    ax.set_aspect(1)
    wb = compute_viewbox(traj.coords, view_par)
    xmin, xmax, ymin, ymax = inflate_viewbox(*wb, 20)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    self.view = DoublePendulumView(view_par)
    for p in self.patches:
      ax.add_patch(p)

  def update(self, t):
    q = self.traj_sp(t)
    self.view.move(q)

  @property
  def patches(self):
    return self.view.patches

def draw(ax, q : np.ndarray, view_par : DoublePendulumViewParam) -> DoublePendulumView:
  q = np.array(q, float)
  ax = plt.gca()
  wb = compute_viewbox(q, view_par)
  xmin, xmax, ymin, ymax = inflate_viewbox(*wb, 10)
  plt.axis('equal')
  # ax.set_aspect('equal', anchor='C')
  ax.set_xlim((xmin, xmax))
  ax.set_ylim((ymin, ymax))
  model = DoublePendulumView(view_par)
  for p in model.patches:
    ax.add_patch(p)
  model.move(q)
  return model

def enlarge_rect(r, coef):
  c = np.mean(r, axis=1)
  w = r[:,1] - r[:,0]
  return np.array([c - w * coef / 2, c + w * coef / 2]).T

def get_traj_bounding_rect(traj : Trajectory):
  qmin = np.min(traj.coords, axis=0)
  qmax = np.max(traj.coords, axis=0)
  return np.array([qmin, qmax]).T

def get_joints_coords(traj : Trajectory, view_par : DoublePendulumViewParam):
  q1, q2 = traj.coords.T
  l1, l2 = view_par.links_length
  r1 = l1 * np.array([np.sin(q1), np.cos(q1)]).T
  r2 = r1 + l2 * np.array([np.sin(q1 + q2), np.cos(q1 + q2)]).T
  return r1, r2

def get_cartesian_rect(traj : Trajectory, view_par : DoublePendulumViewParam):
  q1, q2 = traj.coords.T
  r1, r2 = get_joints_coords(traj, view_par)
  x1, y1 = r1.T
  x2, y2 = r2.T

  xmin = min(0, np.min(x1))
  xmin = min(xmin, np.min(x2))

  xmax = max(0, np.max(x1))
  xmax = max(xmax, np.max(x2))

  ymin = min(0, np.min(y1))
  ymin = min(ymin, np.min(y2))

  ymax = max(0, np.max(y1))
  ymax = max(ymax, np.max(y2))

  return np.array([
    [xmin, xmax],
    [ymin, ymax]
  ])

def motion_schematic(traj : Trajectory, view_par : DoublePendulumViewParam):
  d = traj.phase - traj.phase[0]
  d = np.linalg.norm(d, axis=1)
  i, = np.nonzero(d < 1e-5)
  i = i[1]
  q1 = traj.coords[0]
  q2 = traj.coords[i//4]
  q3 = traj.coords[i//2]

  fig,ax = plt.subplots(1, 1, num=f'schematic at {q1[0]:.2f}, {q1[1]:.2f}', figsize=(6, 4))
  ax.set_aspect(1)
  view_par.alpha = 0.5
  draw(ax, q1, view_par)

  view_par.alpha = 0.5
  draw(ax, q3, view_par)

  view_par.alpha = 1
  draw(ax, q2, view_par)

  r = get_cartesian_rect(traj, view_par)
  xdiap, ydiap = enlarge_rect(r, 1.15)
  ax.set_xlim(*xdiap)
  ax.set_ylim(*ydiap)
  plt.grid(True)

  return fig

def motion_schematic_v2(traj : Trajectory, view_par : DoublePendulumViewParam):
  d = traj.phase - traj.phase[0]
  d = np.linalg.norm(d, axis=1)
  i, = np.nonzero(d < 1e-5)
  i = i[1]
  q_centered = traj.coords[i//4]

  fig, ax = plt.subplots(1, 1, num=f'schematic at {q_centered[0]:.2f}, {q_centered[1]:.2f}', figsize=(6, 4))
  ax.set_aspect(1)
  r1, r2  = get_joints_coords(traj, par)
  plt.plot(r1[:,0], r1[:,1], linewidth=2, ls='--', color='grey')
  plt.plot(r2[:,0], r2[:,1], linewidth=2, ls='--', color='grey')
  draw(ax, q_centered, par, alpha=1, color='#3030C0', linewidth=2)
  r = get_cartesian_rect(traj, view_par)
  xdiap, ydiap = enlarge_rect(r, 1.05)
  ax.set_xlim(*xdiap)
  ax.set_ylim(*ydiap)

  plt.grid(True)
  plt.tight_layout()
  return fig


def draw_link(origin, angle, w, l):
  origin = np.reshape(origin, (2,))
  d = np.array([w/2, 0])
  p = origin - d
  rect = Rectangle(p, w, l, rotation_point=tuple(origin))
  rect.set_angle(angle)
  return rect

def move_link(link, angle=None, origin=None):
  if origin is not None:
    pass

def animate(traj : Trajectory, view_par : DoublePendulumViewParam, 
            fps=60, speedup=1, videopath=None):

  q = traj.coords[0]
  fig, ax = plt.subplots(1, 1, num=f'animation at {q[0]:.2f}, {q[1]:.2f}', figsize=(6, 4))
  double_pend = DoublePendulumAnim(ax, view_par, traj)
  return Animate(fig, [double_pend], traj.time[-1], fps, speedup, videopath)

def animate_with_graphs(traj : Trajectory, view_par : DoublePendulumViewParam, 
                        fps=60, speedup=1, videopath=None):

  fig, axes = plt.subplot_mosaic([['coords', 'anim'],
                                  ['vels', 'anim'],
                                  ['control', 'anim']],
                                 figsize=(6, 4), layout='constrained',
                                 num='animation')

  animators = [
    AnimGraph(axes['coords'], traj.time, traj.coords),
    AnimGraph(axes['vels'], traj.time, traj.vels),
    AnimGraph(axes['control'], traj.time, traj.control),
    DoublePendulumAnim(axes['anim'], view_par, traj),
  ]
  return Animate(fig, animators, traj.time[-1], fps, speedup, videopath)

def test_view():
  fig, ax = plt.subplots(1, 1)
  plt.grid(True, zorder=1)
  ax.set_aspect(1)
  par = double_pendulum_param_default
  pend_view = DoublePendulumView(par)

  for p in pend_view.patches:
    ax.add_patch(p)

  pend_view.move([0.5, -0.3])
  plt.xlim(-2, 2)
  plt.ylim(-2, 2)
  plt.show()

def test_anim():
  par = get_view_parameters(double_pendulum_param_default)
  t = np.linspace(0, 5)
  fig, ax = plt.subplots(1, 1)
  phase = np.array([np.sin(t), np.cos(t), np.zeros(*t.shape), np.zeros(*t.shape)]).T
  traj = Trajectory(time = t, phase = phase)
  anim = DoublePendulumAnim(ax, par, traj)
  plt.show()

if __name__ == '__main__':
  test_anim()
