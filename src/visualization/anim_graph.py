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
  def patches(self):
    return self.lines

class AnimScope:
  def __init__(self, ax, t, x, y, setlims=True, **plot_args):
    self.t = np.array(t)
    if np.ndim(y) == 1:
      npts, = np.shape(y)
      ydim = 1
    elif np.ndim(y) == 2:
      npts, ydim = np.shape(y)
    else:
      assert False, 'Expect y to have 1 or 2 dims'

    assert self.t.shape == (npts,)
    self.x = np.reshape(x, (npts,))
    self.y = np.reshape(y, (npts, ydim))
    self.lines = ax.plot(self.x[0:1], self.y[0:1,:], **plot_args)

    if setlims:
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
    y = self.y[mask,:]

    for i, line in enumerate(self.lines):
      line.set_data(x, y[:,i])

  @property
  def patches(self):
    return tuple(self.lines)

class AnimTrace:
  def __init__(self, ax, t, x, y, lifetime, setlims=True, **plot_args):
    self.t = np.copy(t)
    self.x = np.copy(x)
    self.y = np.copy(y)
    self.lifetime = lifetime
    self.line, = plt.plot(x[0], y[0], **plot_args)

    if setlims:
      xmin = np.min(self.x)
      xmax = np.max(self.x)
      ymin = np.min(self.y)
      ymax = np.max(self.y)

      k = 1.05
      ax.set_xlim(xmax - k * (xmax - xmin), xmin + k * (xmax - xmin))
      ax.set_ylim(ymax - k * (ymax - ymin), ymin + k * (ymax - ymin))

  def update(self, t):
    mask = (self.t <= t) & (self.t > t - self.lifetime)
    x = self.x[mask]
    y = self.y[mask]
    self.line.set_data(x, y)

  @property
  def patches(self):
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
    return graph.patches

  anim = animation.FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True)
  plt.show()

def test_scope():
  fig, ax = plt.subplots(figsize=(6, 4), num='graph sim')
  t = np.linspace(0, 30, 1000)
  x = np.sin(2. * t)
  y1 = np.sin(3.05 * t + 1.)
  y2 = np.sin(3.06 * t + 1.01)
  y = np.array([y1, y2]).T
  scope = AnimScope(ax, t, x, y)
  # scope = AnimScope(ax, t, x, y1)

  fps = 60.
  nframes = int(fps * t[-1])

  def update(i):
    t = i / fps
    scope.update(t)
    return scope.patches

  anim = animation.FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True)
  plt.show()

def test_trace():
  fig, ax = plt.subplots(figsize=(6, 4), num='graph sim')
  t = np.linspace(0, 30, 1000)
  x = np.sin(2. * t)
  y1 = np.sin(3.05 * t + 1.)
  y2 = np.sin(3.06 * t + 1.01)
  y = np.array([y1, y2]).T
  scope = AnimTrace(ax, t, x, y1, 1.)

  fps = 60.
  nframes = int(fps * t[-1])

  def update(i):
    t = i / fps
    scope.update(t)
    return scope.patches

  anim = animation.FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True)
  plt.show()

if __name__ == '__main__':
  test_scope()
  test_trace()
