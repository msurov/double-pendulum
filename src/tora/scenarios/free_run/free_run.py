from tora.dynamics import TORADynamics, TORAPar, tora_param_default
from tora.anim import launch_anim, get_vis_par
from common.mechsys_integ import integrate
import matplotlib.pyplot as plt


def main():
  par = tora_param_default
  sys = TORADynamics(par)
  traj = integrate(sys, [0., 1.0], [0., 0.], 10., max_step=1e-2)
  vispar = get_vis_par(par)
  a = launch_anim(traj, vispar)
  plt.show()

if __name__ == '__main__':
  main()
