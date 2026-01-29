import scienceplots
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import casadi as ca
from common.numpy_utils import map_array
from scipy.optimize import brentq

cos = ca.cos
tan = ca.tan
sin = ca.sin
pi = ca.pi
sqrt = ca.sqrt


def find_all_roots(f, interval):
  x = np.linspace(*interval, 100)
  y = map_array(f, x, 1)
  mask = y[1:] * y[:-1] <= 0
  indices, = np.nonzero(mask)
  roots = []
  for i in indices:
    r = brentq(f, x[i], x[i+1], maxiter=40, xtol=1e-8)
    roots.append(r)

  return roots

def main():
  l = 1
  nu = 0.0
  g = 9.8

  theta = ca.SX.sym('theta')

  q = 3
  # a1 = ca.SX.sym('a1')
  a1 = 0.1
  theta_s = 0.4
  ddX = 6.
  a2 = -1 / q * (a1 + l / cos(theta_s))
  b2 = -2.
  b1 = -ddX - b2 * q**2
  X = a1 * sin(theta - theta_s) + b1 * cos(theta - theta_s) + \
    a2 * sin(q * theta - q * theta_s) + b2 * cos(q * theta - q * theta_s)
  dX = ca.jacobian(X, theta)
  ddX = ca.jacobian(dX, theta)
  
  alpha = cos(theta) * dX + l
  dalpha = ca.jacobian(alpha, theta)
  beta = cos(theta) * ddX
  gamma = -g * sin(theta)
  delta = nu

  alpha_fun = ca.Function('alpha', [theta], [alpha])
  beta_fun = ca.Function('beta', [theta], [beta])
  gamma_fun = ca.Function('gamma', [theta], [gamma])
  dalpha_fun = ca.Function('gamma', [theta], [dalpha])

  singularities = find_all_roots(alpha_fun, [0, 2*np.pi])
  for theta_s in singularities:
    print('--')
    print('theta_s = ', theta_s)
    print('alpha = ', alpha_fun(theta_s))
    print('dalpha = ', dalpha_fun(theta_s))
    print('gamma / beta = ', gamma_fun(theta_s) / beta_fun(theta_s))

  return

  y_s = (-delta + sqrt(delta**2 - 4 * beta * gamma)) / (2 * beta)
  y_s = ca.substitute(y_s, theta, theta_s)

  y = ca.SX.sym('y')  
  f = -beta * y - delta - gamma / y
  f_theta = ca.jacobian(f, theta)
  f_y = ca.jacobian(f, y)
  dy_s = f_theta / (dalpha - f_y)
  dy_s = ca.substitute(dy_s, theta, theta_s)
  dy_s = ca.substitute(dy_s, y, y_s)

  y = ca.SX.sym('y')
  dy = -(beta * y + delta + gamma / y) / alpha

  a1_val = -0.8
  eps = 0.01
  theta0 = theta_s + eps
  tmp = ca.substitute(dy, a1, a1_val)
  dy_fun = ca.Function('rhs', [theta, y], [tmp])
  rhs = lambda theta, y: np.reshape(dy_fun(theta, y), (1,))

  dy_s_val = ca.evalf(ca.substitute(dy_s, a1, a1_val))
  y0 = y_s + dy_s_val * (theta0 - theta_s)
  y0 = float(y0)

  # sol = solve_ivp(rhs, [theta0, theta0 + np.pi - 0.4], [y0], max_step=1e-2)
  # plt.plot(sol.t, sol.y[0])

  # plt.xlabel(R'$\theta$')
  # plt.ylabel(R'$\dot \theta$')
  # plt.tight_layout(pad=0.1)
  # plt.show()

if __name__ == '__main__':
  plt.style.use('science')
  plt.rcParams['legend.frameon'] = True
  plt.rcParams['legend.framealpha'] = 0.8
  main()
