from anim import draw
import matplotlib.pyplot as plt
from double_pendulum.dynamics.parameters import DoublePendulumParam, load


def save_image():
  fig,ax = plt.subplots(1,1)
  q = [0.7, -0.4]
  par = load('src/config/parameters.json')
  plt.grid(True)
  draw(q, par)
  plt.savefig('./doc/double-pendulum.svg')
  plt.show()

if __name__ == '__main__':
  save_image()
