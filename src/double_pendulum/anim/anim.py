import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np
from scipy.interpolate import make_interp_spline
from double_pendulum.dynamics import DoublePendulumParam
from common.trajectory import Trajectory

class DoublePendulumAnim:
  def __init__(self, par : DoublePendulumParam, color='black', linewidth=3, **plotargs):
    self.line, = plt.plot([0,0,0], [0,1,2], '-o', linewidth=linewidth, color=color, **plotargs)
    self.l1, self.l2 = par.lengths

  def move(self, q):
    x1 = self.l1 * np.sin(q[0])
    y1 = self.l1 * np.cos(q[0])
    x2 = x1 + self.l2 * np.sin(q[0] + q[1])
    y2 = y1 + self.l2 * np.cos(q[0] + q[1])
    x = np.array([0, x1, x2])
    y = np.array([0, y1, y2])
    self.line.set_data(x, y)

  def elems(self):
    return self.line,

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

def draw(q : np.ndarray, par : DoublePendulumParam, **plot_args):
  q = np.array(q, float)
  ax = plt.gca()
  wb = compute_viewbox(q, par)
  xmin, xmax, ymin, ymax = inflate_viewbox(*wb, 10)
  plt.axis('equal')
  ax.set_xlim((xmin, xmax))
  ax.set_ylim((ymin, ymax))
  model = DoublePendulumAnim(par, **plot_args)
  model.move(q)
  return model.elems()

def animate(traj : Trajectory, par : DoublePendulumParam, fps=60, speedup=1):
  q = traj.coords
  t = traj.time
  qfun = make_interp_spline(t, q, k=1)

  wb = compute_viewbox(q, par)
  xmin, xmax, ymin, ymax = inflate_viewbox(*wb, 10)

  fig, ax = plt.subplots(figsize=(4 * (xmax - xmin) / (ymax - ymin), 4))
  plt.gca().set(xlim=[xmin, xmax], ylim=[ymin, ymax])
  plt.gca().set_aspect(1)

  model = DoublePendulumAnim(par)

  animtime = (t[-1] - t[0]) / speedup
  nframes = int(animtime * fps)

  def drawframe(iframe):
    ti = speedup * iframe / fps + t[0]
    model.move(qfun(ti))
    return model.elems()

  anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=1000/fps, blit=True)
  rc('animation', html='jshtml')
  return anim

"""
def animate2(traj : Trajectory, par : DoublePendulumAnim, fps=60):
  q = traj.coords
  t = traj.time
  qfun = make_interp_spline(t, q, k=1)

  wb = compute_viewbox(q, par)
  xmin, xmax, ymin, ymax = inflate_viewbox(*wb, 10)

  fig, axes = plt.subplots(1, 2, figsize=(8 * (xmax - xmin) / (ymax - ymin), 4))
  plt.sca(axes[0])
  plt.axis('equal')
  axes[0].set_ylim((ymin, ymax))
  axes[0].set_xlim((xmin, xmax))
  model = DoublePendulumAnim(par)
  interval = t[-1] - t[0]
  nframes = int(interval * fps)

  def drawframe(iframe):
    ti = iframe / fps + t[0]
    model.move(qfun(ti))
    return model.elems()

  anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=1000/fps, blit=True)
  rc('animation', html='jshtml')
  return anim
"""