import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp
from common.lqr import lqr_ltv_periodic


def ltv_lqr_periodic_test():

  def A(t):
    return np.array([
      [ 1 * np.sin( 3 * t + 0),  1 * np.sin( 2 * t + 2),  8 * np.sin( 1 * t + 4)],
      [                      1,                       0,  3 * np.sin( 4 * t - 1)],
      [-3 * np.sin(-1 * t + 2), -3 * np.sin( 3 * t + 2),                       0],
    ])

  def B(t):
    return np.array([
      [                      0,                          1],
      [                      0, 2 + 5 * np.cos(-1 * t + 2)],
      [-2 * np.sin(-3 * t + 2), 0 - 3 * np.sin( 1 * t + 5)],
    ])

  Q = np.eye(3)
  R = np.eye(2)
  t = np.linspace(0, 2*np.pi, 100)
  K, P = lqr_ltv_periodic(t, A, B, Q, R, max_step=1e-2)
  K = make_interp_spline(t, K, k=3)

  def rhs(t, x):
    return A(t) @ x + B(t) @ K(t) @ x

  np.random.seed(1)
  x0 = np.random.normal(size=3)
  sol = solve_ivp(rhs, [0, 2*np.pi], x0, max_step=1e-2)
  assert np.allclose(sol.y[:,-1], 0, atol=1e-5, rtol=0)
