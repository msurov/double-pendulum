from typing import Tuple
from cart_pendulum.dynamics import CartPendulumDynamics, CartPendulumParam, cart_pendulum_param_default
from cart_pendulum.anim import CartPendulumVis, get_vis_par, launch_anim
from common.mechsys_integ import integrate, simulate
import matplotlib.pyplot as plt
import casadi as ca
import numpy as np
from sigproc.delay_filter import DelayFilter


def soft_sign(x, eps=1e-5):
  return x / (np.abs(x) + eps)

class Feedback:
  def __init__(self, sys : CartPendulumDynamics, q0 : Tuple[float, float], fbcoef : float):
    w = ca.MX.sym('w')
    B = sys.B_expr
    Bpinv = B.T
    Minv = ca.pinv(sys.M_expr)
    C = sys.C_expr
    dq = sys.dq
    G = sys.G_expr
    u_expr = (w + Bpinv @ Minv @ (C @ dq + G)) / (Bpinv @ Minv @ B)
    u_fun = ca.Function('u', [sys.q, sys.dq, w], [u_expr])

    self.fb_transform_fun = u_fun
    self.fbcoef = fbcoef

    self.kE = 5.
    self.period = 2 * np.pi / np.sqrt(fbcoef)
    self.__delay_filter = DelayFilter(2 * self.period)

  def __call__(self, t, q, dq):
    E = dq[0]**2 / 2 + self.fbcoef * q[0]**2 / 2
    if not hasattr(self, 'E'):
      self.E = E
    E_stab = -self.kE * (E - self.E) * soft_sign(dq[0])
    w = -self.fbcoef * q[0] + E_stab
    u = self.fb_transform_fun(q, dq, w)
    state_dalayed = self.__delay_filter(t, np.concatenate((q, dq)))
    if t < self.__delay_filter.delay:
      stab = 0.
    else:
      stab = 0. * (dq[1] - state_dalayed[3])
    u += stab
    return float(u)

def main():
  par = cart_pendulum_param_default
  sys = CartPendulumDynamics(par)
  # q0 = np.array([0.6, 1.87155])
  # q0 = np.array([0.6, 1.88])
  q0 = np.array([0.490657, 1.99031])
  q0 = np.array([0.704495, 1.80434])
  fbcoef = 30.
  fb = Feedback(sys, q0, fbcoef)

  traj = simulate(sys, q0, [0., 0.], 1e-3, 5 * fb.period, fb)

  plt.figure('x')
  plt.plot(traj.time, traj.coords[:,0])
  plt.plot(traj.time, q0[0] * np.cos(traj.time * np.sqrt(fbcoef)))
  plt.grid(True)

  plt.figure('theta')
  plt.plot(traj.coords[:,1], traj.vels[:,1])
  plt.grid(True)

  a = launch_anim(traj, get_vis_par(par))
  plt.show()


if __name__ == '__main__':
  main()
