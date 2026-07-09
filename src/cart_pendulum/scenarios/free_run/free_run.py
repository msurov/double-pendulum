from cart_pendulum.dynamics import CartPendulumDynamics, CartPendulumPar, cart_pendulum_param_default
from cart_pendulum.anim import CartPendulumVis, get_vis_par, launch_anim
from common.mechsys_integ import integrate
import matplotlib.pyplot as plt


def main():
  par = cart_pendulum_param_default
  par.joint_friction = 0.1
  sys = CartPendulumDynamics(par)
  traj = integrate(sys, [0., 0.1], [0., 0.], 10., max_step=1e-2)
  vispar = get_vis_par(par)
  a = launch_anim(traj, vispar)
  plt.show()

if __name__ == '__main__':
  main()
