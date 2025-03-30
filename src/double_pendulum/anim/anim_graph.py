import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np


class AnimGraph:
  def __init__(self, ax, t, y):
    self.t = np.array(t)
    self.y = np.array(y)
    if self.y.ndim == 1:
      self.y = self.y[:,np.newaxis]
    assert self.y.ndim == 2
    self.lines = ax.plot(self.t[0:1], self.y[0:1,...])
    self.lines = tuple(self.lines)

    tmin = np.min(self.t)
    tmax = np.max(self.t)
    ymin = np.min(self.y)
    ymax = np.max(self.y)

    ax.set_xlim(tmin, tmax)
    ax.set_ylim(ymin, ymax)

  def update(self, t):
    mask = self.t <= t
    tarr = self.t[mask]
    yarr = self.y[mask,...]
    for line, y in zip(self.lines, yarr.T):
      line.set_data(tarr, y)
  
  @property
  def elems(self):
    return self.lines

class AnimScope:
  def __init__(self, ax, t, x, y, **plot_args):
    self.t = np.array(t)
    npts, = self.t.shape
    self.x = np.reshape(x, (npts,))
    self.y = np.reshape(y, (npts,))
    self.line, = ax.plot(self.x[0:1], self.y[0:1], **plot_args)

    xmin = np.min(self.x)
    xmax = np.max(self.x)
    ymin = np.min(self.y)
    ymax = np.max(self.y)

    k = 1.05
    ax.set_xlim(xmax - k * (xmax - xmin), xmin + k * (xmax - xmin))
    ax.set_ylim(ymax - k * (ymax - ymin), ymin + k * (ymax - ymin))

  def update(self, t):
    mask = self.t <= t
    x = self.x[mask]
    y = self.y[mask]
    self.line.set_data(x, y)

  @property
  def elems(self):
    return self.line,

def test_graph():
  fig, ax = plt.subplots(figsize=(6, 4), num='graph sim')
  t = np.linspace(0, 10, 1000)
  y = np.array([np.sin(2. * t), np.sin(2. * t + 1.)]).T
  graph = AnimGraph(ax, t, y)

  fps = 60.
  nframes = int(fps * t[-1])

  def update(i):
    t = i / fps
    graph.update(t)
    return graph.elems

  anim = animation.FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True)
  plt.show()

def test_scope():
  fig, ax = plt.subplots(figsize=(6, 4), num='graph sim')
  t = np.linspace(0, 30, 1000)
  xy = np.array([np.sin(2. * t), np.sin(3.05 * t + 1.)]).T
  scope = AnimScope(ax, t, xy)

  fps = 60.
  nframes = int(fps * t[-1])

  def update(i):
    t = i / fps
    scope.update(t)
    return scope.elems

  anim = animation.FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True)
  plt.show()

if __name__ == '__main__':
  test_scope()
