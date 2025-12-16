from typing import Tuple
from cart_pendulum.dynamics import CartPendulumDynamics, CartPendulumParam, cart_pendulum_param_default
from cart_pendulum.anim import CartPendulumVis, get_vis_par, launch_anim
import matplotlib.pyplot as plt
import numpy as np
from sigproc.delay_filter import DelayFilter
import casadi as ca
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt


def rk4_step(f, t, x, h):
  k1 = f(x)
  k2 = f(x + 0.5*h*k1)
  k3 = f(x + 0.5*h*k2)
  k4 = f(x + h*k3)
  return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(sys : Callable[[float, np.ndarray], np.ndarray], x0 : np.ndarray, step : float, nsteps : int) -> Tuple[np.ndarray, np.ndarray]:
  x = x0
  t = 0.
  x_arr = [x]
  t_arr = [t]
  for i in range(1, nsteps + 1):
    x = rk4_step(sys, t, x, step)
    t += step
    x_arr.append(x)
    t_arr.append(t)
  return t_arr, x_arr

def get_reduced_system(sys : CartPendulumDynamics, fbcoef):
  B_perp = sys.B_perp_expr
  M = sys.M_expr
  C = sys.C_expr
  dq = sys.dq
  G = sys.G_expr
  ddx = -fbcoef * sys.q[0]
  tmp = B_perp @ M
  m1 = tmp[0,0]
  m2 = tmp[0,1]
  ddtheta = (-B_perp @ C @ dq - B_perp @ G - m1 * ddx) / m2
  rhs = ca.vertcat(dq, ddx, ddtheta)
  state = ca.vertcat(sys.q, dq)
  rhs_fun = ca.Function('rhs', [state], [rhs])
  return rhs_fun

def solve_numerically():
  par = cart_pendulum_param_default
  sys = CartPendulumDynamics(par)
  fbcoef = 30.
  reduced = get_reduced_system(sys, fbcoef)
  period = 2 * np.pi / np.sqrt(fbcoef)
  integtime = period
  nsteps = 300
  step = integtime / nsteps
  q0 = ca.DM([0.6, 1.87155])
  dq0 = ca.DM.zeros(2)
  x0 = ca.vertcat(q0, dq0)
  t, x = integrate(reduced, x0, step, nsteps)
  x = np.array(x)[:,:,0]
  plt.plot(x[:,0], x[:,2])
  plt.show()

def main():
  par = cart_pendulum_param_default
  sys = CartPendulumDynamics(par)
  fbcoef = 30.
  reduced = get_reduced_system(sys, fbcoef)
  period = 2 * np.pi / np.sqrt(fbcoef)
  integtime = period / 2
  nsteps = 300
  step = integtime / nsteps
  q0 = ca.MX.sym('q0', 2)
  dq0 = ca.MX.zeros(2)
  x0 = ca.vertcat(q0, dq0)
  t, x = integrate(reduced, x0, step, nsteps)
  xf = x[-1]
  xf_ = ca.substitute(xf, q0, [0.490657, 1.99031])
  xf_ = ca.evalf(xf_)

  nlp = {
    'x': q0,
    'f': ca.dot(xf[2:4], xf[2:4])
  }
  solver = ca.nlpsol('solver', 'ipopt', nlp)

  initial_guess = np.array([0.5, 1.57])
  solution = solver(x0=initial_guess)
  print('q0:', solution['x'])

if __name__ == '__main__':
  main()
