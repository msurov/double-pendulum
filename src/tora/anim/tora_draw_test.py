from tora.anim import ToraVisPar, ToraVis
import matplotlib.pyplot as plt


def main():
  par = ToraVisPar(
    cart_width = 0.4,
    cart_height = 0.16,
    rod_length = 1.,
    rod_width = 0.03,
    wheel_radius = 0.05,
  )
  fig, ax = plt.subplots(1, 1)
  ax.set_aspect(1)
  vis = ToraVis(ax, par)
  vis.move([0.1, -1.0])
  plt.xlim(-1, 1)
  plt.ylim(-1.2, 1.2)
  plt.tight_layout(pad = 0.1, h_pad = 0.1)
  plt.show()

if __name__ == '__main__':
  main()
