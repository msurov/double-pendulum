import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from pvtol.anim.matplotlib_svg import load_path_collection


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

def draw(ax, q, size=1):
  model = PVTOLView(ax, size=size)
  model.move(q)
  return model

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
