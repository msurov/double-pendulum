import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from .matplotlib_svg import load_path_collection


class PVTOLView:
  def __init__(self, ax, size=1):
    self.transform1 = Affine2D() \
      .translate(-100, -50) \
      .rotate_deg(180) \
      .scale(size / 160)
    model = load_path_collection('fig/pvtol.svg')
    ax.add_collection(model)
    self.transform2 = model.get_transform()
    model.set_transform(self.transform1 + self.transform2)
    self.model = model

  def move(self, q):
    psi = q[2]
    x = q[0]
    z = q[1]
    t = Affine2D().rotate(psi).translate(x, z)
    self.model.set_transform(self.transform1 + t + self.transform2)
  
  @property
  def elems(self):
    return self.model,

def draw(ax, q, size=1):
  model = PVTOLView(ax, size=size)
  model.move(q)
  return model

def expand_box(xmin, xmax, ymin, ymax, pcnt):
  xc = (xmax + xmin) / 2
  w = (xmax - xmin) * (100 + pcnt) / 100
  yc = (ymax + ymin) / 2
  h = (ymax - ymin) * (100 + pcnt) / 100
  return (
    xc - w / 2,
    xc + w / 2,
    yc - h / 2,
    yc + h / 2,
  )

def compute_occupancy_box(q_arr, vtol_size):
  pts = []
  for q in q_arr:
    p1 = np.array([
      q[0] + np.cos(q[2]) * 0.5 * vtol_size,
      q[1] + np.sin(q[2]) * 0.5 * vtol_size,
    ])
    p2 = np.array([
      q[0] - np.cos(q[2]) * 0.5 * vtol_size,
      q[1] - np.sin(q[2]) * 0.5 * vtol_size,
    ])
    pts.append(p1)
    pts.append(p2)
  pts = np.array(pts)
  print(pts)
  xmin,ymin = np.min(pts, axis=0)
  xmax,ymax = np.max(pts, axis=0)
  return xmin, xmax, ymin, ymax

def test():
  fig = plt.figure()
  plt.axis('equal')
  ax = plt.gca()
  draw(ax, [0.32, 0.33, np.pi/2])
  ax.set_xlim(-1, 2)
  plt.grid(True)
  plt.show()

if __name__ == '__main__':
  test()
