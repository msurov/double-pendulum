from cart_nlinks_pendulum.dynamics import (
  CartNLinksPendDynamics,
  CartNLinksPendPar,
  cart_nlinks_pend_par_default
)
from cart_nlinks_pendulum.anim import get_vis_par, CartNLinksPendVis, launch_anim
from common.mechsys_integ import integrate
import numpy as np
import matplotlib.pyplot as plt


def main():
  par = cart_nlinks_pend_par_default  
  sys = CartNLinksPendDynamics(par)
  np.random.seed(0)
  q0 = 1e-2 * np.random.normal(size=4)
  dq0 = np.zeros(4)
  traj = integrate(sys, q0, dq0, 20., max_step=1e-2)
  
  a = launch_anim(traj, par)
  plt.show()

if __name__ == '__main__':
  main()
