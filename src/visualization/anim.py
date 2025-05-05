from matplotlib import animation
from functools import reduce
import operator


class Animate:
  def __init__(self, fig, animators, animation_time, fps=30., speedup=1., dpi=90, videopath=None):

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
      self.anim.save(videopath, fps=fps, dpi=dpi)

  def update(self, t):
    for a in self.animators:
      a.update(t)

  @property
  def elems(self):
    elems = reduce(operator.add, (a.elems for a in self.animators), tuple())
    elems = tuple(elems)
    return elems
