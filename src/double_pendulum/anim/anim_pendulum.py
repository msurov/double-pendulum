import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from double_pendulum.dynamics import DoublePendulumParam
from common.trajectory import Trajectory

def compute_viewbox(q : np.ndarray, pr : DoublePendulumParam):
  theta1 = q[...,0]
  theta2 = q[...,1]
  l1,l2 = pr.lengths

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

class DoublePendulumView:
  def __init__(self, par : DoublePendulumParam, color='black', linewidth=8, markersize=12, **plotargs):
    self.shadow, = plt.plot([0,0,0], [0,1,2], '-o', 
                            linewidth=1.5*linewidth, markersize=1.5*markersize, 
                            color='#F0F0F0', **plotargs)
    self.line, = plt.plot([0,0,0], [0,1,2], '-o', 
                          linewidth=linewidth, markersize=markersize, 
                          color=color, **plotargs)
    self.l1, self.l2 = par.lengths

  def move(self, q):
    x1 = self.l1 * np.sin(q[0])
    y1 = self.l1 * np.cos(q[0])
    x2 = x1 + self.l2 * np.sin(q[0] + q[1])
    y2 = y1 + self.l2 * np.cos(q[0] + q[1])
    x = np.array([0, x1, x2])
    y = np.array([0, y1, y2])
    self.shadow.set_data(x, y)
    self.line.set_data(x, y)

  def elems(self):
    return self.shadow, self.line,

class DoublePendulumAnim:
  def __init__(self, ax, par : DoublePendulumParam, traj : Trajectory, 
                shadow_color = '#F0F0F0', 
                vertsize = 12, linkwidth = 4, link_color = 'darkblue'):

    self.l1, self.l2 = par.lengths
    self.traj_sp = make_interp_spline(traj.time, traj.coords, k=1)
    self.traj_sp.extrapolate = None
    q = self.traj_sp(0.)
    x, y = self.solve_forward_kinematics(q)
    self.shadow, = ax.plot(x, y, '-o', linewidth=2.5*linkwidth,
                            markersize=1.5*vertsize, color=shadow_color)
    self.line, = ax.plot(x, y, '-o', linewidth=linkwidth,
                          markersize=vertsize, color=link_color)

    ax.set_aspect(1)
    wb = compute_viewbox(traj.coords, par)
    xmin, xmax, ymin, ymax = inflate_viewbox(*wb, 10)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

  def solve_forward_kinematics(self, q : np.ndarray):
    x1 = self.l1 * np.sin(q[0])
    y1 = self.l1 * np.cos(q[0])
    x2 = x1 + self.l2 * np.sin(q[0] + q[1])
    y2 = y1 + self.l2 * np.cos(q[0] + q[1])
    x = np.array([0, x1, x2])
    y = np.array([0, y1, y2])
    return x, y

  def update(self, t):
    q = self.traj_sp(t)
    x, y = self.solve_forward_kinematics(q)
    self.shadow.set_data(x, y)
    self.line.set_data(x, y)

  @property
  def elems(self):
    return self.shadow, self.line

def draw(q : np.ndarray, par : DoublePendulumParam, **plot_args):
  q = np.array(q, float)
  ax = plt.gca()
  wb = compute_viewbox(q, par)
  xmin, xmax, ymin, ymax = inflate_viewbox(*wb, 10)
  plt.axis('equal')
  # ax.set_aspect('equal', anchor='C')
  ax.set_xlim((xmin, xmax))
  ax.set_ylim((ymin, ymax))
  model = DoublePendulumView(par, **plot_args)
  model.move(q)
  return model.elems()

def enlarge_rect(r, coef):
  c = np.mean(r, axis=1)
  w = r[:,1] - r[:,0]
  return np.array([c - w * coef / 2, c + w * coef / 2]).T

def get_traj_bounding_rect(traj : Trajectory):
  qmin = np.min(traj.coords, axis=0)
  qmax = np.max(traj.coords, axis=0)
  return np.array([qmin, qmax]).T

def get_joints_coords(traj : Trajectory, par : DoublePendulumParam):
  q1, q2 = traj.coords.T
  l1,l2 = par.lengths
  r1 = l1 * np.array([np.sin(q1), np.cos(q1)]).T
  r2 = r1 + l2 * np.array([np.sin(q1 + q2), np.cos(q1 + q2)]).T
  return r1, r2

def get_cartesian_rect(traj : Trajectory, par : DoublePendulumParam):
  q1, q2 = traj.coords.T
  r1, r2 = get_joints_coords(traj, par)
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

def motion_schematic(traj : Trajectory, par : DoublePendulumParam):
  d = traj.phase - traj.phase[0]
  d = np.linalg.norm(d, axis=1)
  i, = np.nonzero(d < 1e-5)
  i = i[1]
  q1 = traj.coords[0]
  q2 = traj.coords[i//4]
  q3 = traj.coords[i//2]

  fig,ax = plt.subplots(1, 1, num=f'schematic at {q1[0]:.2f}, {q1[1]:.2f}', figsize=(6, 4))
  ax.set_aspect(1)
  draw(q1, par, alpha=1, color='#3030E0', linewidth=4)
  draw(q2, par, alpha=1, color='#3030B0', linewidth=4)
  draw(q3, par, alpha=1, color='#303080', linewidth=4)
  r = get_cartesian_rect(traj, par)
  xdiap, ydiap = enlarge_rect(r, 1.05)
  ax.set_xlim(*xdiap)
  ax.set_ylim(*ydiap)

  plt.grid(True)
  plt.tight_layout()
  return fig

def motion_schematic_v2(traj : Trajectory, par : DoublePendulumParam):
  d = traj.phase - traj.phase[0]
  d = np.linalg.norm(d, axis=1)
  i, = np.nonzero(d < 1e-5)
  i = i[1]
  q_centered = traj.coords[i//4]

  fig,ax = plt.subplots(1, 1, num=f'schematic at {q_centered[0]:.2f}, {q_centered[1]:.2f}', figsize=(6, 4))
  ax.set_aspect(1)
  r1, r2  = get_joints_coords(traj, par)
  plt.plot(r1[:,0], r1[:,1], linewidth=2, ls='--', color='grey')
  plt.plot(r2[:,0], r2[:,1], linewidth=2, ls='--', color='grey')
  draw(q_centered, par, alpha=1, color='#3030C0', linewidth=2)
  r = get_cartesian_rect(traj, par)
  xdiap, ydiap = enlarge_rect(r, 1.05)
  ax.set_xlim(*xdiap)
  ax.set_ylim(*ydiap)

  plt.grid(True)
  plt.tight_layout()
  return fig
