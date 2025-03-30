from common.trajectory import Trajectory
from double_pendulum.dynamics import DoublePendulumParam
from .anim_graph import AnimGraph
from .anim_pendulum import DoublePendulumAnim
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from functools import reduce
import operator


class Animate:
  def __init__(self, fig, animators, animation_time, fps=30., speedup=1., videopath=None):

    assert animation_time > 0
    assert fps > 0
    assert speedup > 0

    nframes = int(fps * animation_time / speedup)
    self.animators = animators

    def drawframe(iframe):
      t = speedup * iframe / fps
      self.update(t)
      return self.elems

    self.anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=1000/fps, blit=True)
    if videopath:
      self.anim.save(videopath, fps=fps)

  def update(self, t):
    for a in self.animators:
      a.update(t)

  @property
  def elems(self):
    elems = reduce(operator.add, (a.elems for a in self.animators), tuple())
    elems = tuple(elems)
    return elems


def animate(traj : Trajectory, par : DoublePendulumParam, fps=60, speedup=1, videopath=None):
  q = traj.coords[0]
  fig, ax = plt.subplots(1, 1, num=f'animation at {q[0]:.2f}, {q[1]:.2f}', figsize=(6, 4))
  double_pend = DoublePendulumAnim(ax, par, traj)
  return Animate(fig, [double_pend], traj.time[-1], fps, speedup, videopath)

def animate_with_graphs(traj : Trajectory, par : DoublePendulumParam, fps=60, speedup=1, videopath=None):
  fig, axes = plt.subplot_mosaic([['coords', 'anim'],
                                  ['vels', 'anim'],
                                  ['control', 'anim']],
                                 figsize=(6, 4), layout='constrained',
                                 num='animation')

  animators = [
    AnimGraph(axes['coords'], traj.time, traj.coords),
    AnimGraph(axes['vels'], traj.time, traj.vels),
    AnimGraph(axes['control'], traj.time, traj.control),
    DoublePendulumAnim(axes['anim'], par, traj),
  ]
  return Animate(fig, animators, traj.time[-1], fps, speedup, videopath)
