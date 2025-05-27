from pvtol.anim.draw import draw
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


def rotate(vec, angle):
  x, y = vec
  s = np.sin(angle)
  c = np.cos(angle)
  return np.array([x * c - y * s, x * s + y * c])

def plot_generalized_coordinates():
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })

  fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
  ax.set_aspect(1)
  ax.axis('off')

  psi = np.pi/5
  center = np.array([0.4, 0.3])

  v1 = 0.2 * np.array([1.0, 0.0])
  v2 = rotate(v1, psi)
  p1 = center + 1.1 * v1
  p2 = center + v2
  p2 -= 0.05 * (p2 - p1)

  plt.plot([center[0], center[0]], [-0.01, center[1]], ls='--', color='grey', zorder=-1)
  plt.plot([-0.01, center[0] * 1.8], [center[1], center[1]], ls='--', color='grey', zorder=-1)

  psi_angle_arc = FancyArrowPatch(
    p1,
    p2,
    connectionstyle="arc3,rad=0.2",
    arrowstyle="Simple, tail_width=1, head_width=7, head_length=10",
    color="#404040"
  )
  ax.add_patch(psi_angle_arc)

  xaxis = FancyArrowPatch(
    [0, 0],
    [0.8, 0],
    connectionstyle="arc3,rad=0",
    arrowstyle="Simple, tail_width=1, head_width=7, head_length=10",
    color="black"
  )
  ax.add_patch(xaxis)

  yaxis = FancyArrowPatch(
    [0, 0],
    [0, 0.5],
    connectionstyle="arc3,rad=0",
    arrowstyle="Simple, tail_width=1, head_width=7, head_length=10",
    color="black"
  )
  ax.add_patch(yaxis)

  plt.plot([0], [0], 'o', color='black')

  p = (p1 + p2) / 2 + np.array([0.04, 0.06])
  plt.annotate(
    R'$\psi$',
    xy = p,
    xytext = p,
    ha = 'left', va = 'top', 
    xycoords = 'data',
    textcoords = 'data',
    fontsize = 40,
  )

  p = [center[0] - 0.02, -0.01]
  plt.annotate(
    R'$x$',
    xy = p,
    xytext = p, 
    ha = 'left', va = 'top', 
    xycoords = 'data',
    textcoords = 'data',
    fontsize = 40,
  )

  p = [-0.05, center[1] + 0.03]
  plt.annotate(
    R'$z$',
    xy = p,
    xytext = p, 
    ha = 'left', va = 'top', 
    xycoords = 'data',
    textcoords = 'data',
    fontsize = 40,
  )

  draw(ax, [*center, psi])

  plt.xlim(-0.1, 0.85)
  plt.ylim(-0.04, 0.6)
  plt.tight_layout(pad=-0.5)

  return fig

def main():
  fig = plot_generalized_coordinates()
  plt.savefig('fig/pvtol_gen_coords.svg')
  plt.show()

main()
