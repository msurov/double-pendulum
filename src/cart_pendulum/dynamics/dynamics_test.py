from cart_pendulum.dynamics import CartPendulumDynamics, CartPendulumParam, cart_pendulum_param_default
from common.mechsys_integ import integrate
import matplotlib.pyplot as plt


def main():
  par = cart_pendulum_param_default
  sys = CartPendulumDynamics(par)
  traj = integrate(sys, [0., 0.1], [0., 0.], 5., max_step=1e-2)
  plt.plot(traj.time, traj.coords)  
  plt.show()

if __name__ == '__main__':
  main()
