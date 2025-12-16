from abc import ABC
from matplotlib import animation
from functools import reduce
import operator
from typing import List, Tuple
from matplotlib.artist import Artist


class AnimatorObject(ABC):
  patches : Tuple[Artist]
  def update(self, t : float):
    pass

class Animate(AnimatorObject):
  def __init__(self, fig, animators : List[AnimatorObject], animation_time, fps=30., speedup=1., dpi=90, videopath=None):

    assert animation_time > 0
    assert fps > 0
    assert speedup > 0

    nframes = int(fps * animation_time / speedup)
    self.__animators = animators

    def drawframe(iframe):
      t = speedup * iframe / fps
      self.update(t)
      return self.patches

    self.__anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=1000/fps, blit=True)
    if videopath:
      self.__anim.save(videopath, fps=fps, dpi=dpi)

  def update(self, t):
    for a in self.__animators:
      a.update(t)

  @property
  def patches(self):
    patches = reduce(lambda acc, obj: acc + obj.patches, self.__animators, tuple())
    patches = tuple(patches)
    return patches
