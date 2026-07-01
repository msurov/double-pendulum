from ball_and_beam.anim import BallAndBeamVisPar, BallAndBeamVis, get_view_box
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from common.geom_utils import Rect, covering_rect


def main():
  vispar = BallAndBeamVisPar(
    ball_radius = 0.1,
    beam_thickness = 0.1,
    beam_length = 1.0,
    surface_vertical_displacement = 0.20,
    joint_radius = 0.05
  )
  fig, ax = plt.subplots(1, 1, figsize=(5, 4))
  ax.set_aspect('equal')
  babvis = BallAndBeamVis(ax, vispar)

  t = np.arange(0, 50, 0.05)
  theta = 0.5 * np.sin(t + 0.5)
  s = 0.2 + 0.3 * np.sin(t)
  q = np.array([
    theta, s
  ]).T

  box = reduce(lambda rect, q: covering_rect(get_view_box(q, vispar), rect), q, Rect(0, 0, 0, 0))
  print(box)

  plt.xlim(box.x1, box.x2)
  plt.ylim(box.y1, box.y2)
  plt.tight_layout()

  for qi in q:
    babvis.move(qi)
    plt.pause(0.01)
    if not plt.get_fignums():
      break

if __name__ == '__main__':
  main()
