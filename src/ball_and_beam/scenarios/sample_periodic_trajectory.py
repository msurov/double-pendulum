import matplotlib.pyplot as plt
from common.trajectory import Trajectory, make_traj
from ball_and_beam.dynamics import BallAndBeamDynamics, ball_and_beam_parameters_default
import casadi as ca
from common.numpy_utils import map_array, find_all_roots
from common.mechsys import MechanicalSystem
from common.trajectory import Trajectory, make_traj
import numpy as np
from common.plots import set_pi_xticks
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from singular_motion_planner.reduced_dynamics import (
  reconstruct_trajectory,
  ReducedDynamics
)
from typing import Optional, Tuple
from ball_and_beam.anim import launch_anim, get_vis_par, BallAndBeamVisPar


def get_singularities(reduced : ReducedDynamics):
  sings = find_all_roots(reduced.alpha, [0, 2 * np.pi])
  result = []
  for sing in sings:
    poly = [
      float(reduced.beta(sing)),
      float(reduced.delta(sing)),
      float(reduced.gamma(sing))
    ]
    roots = np.roots(poly)
    real_roots = np.abs(np.imag(roots)) < 1e-12
    roots = np.real(roots[real_roots])
    roots.sort()
    result.append(
      (sing, *roots)
    )

  return result

def get_equilibriums(reduced : ReducedDynamics):
  equils = find_all_roots(reduced.gamma, [0, 2 * np.pi])
  return equils

def get_accel_at_sing(x_sing, dx_sing, reduced : ReducedDynamics):
  arg = reduced.s

  alpha = reduced.alpha_expr
  beta = reduced.beta_expr
  delta = reduced.delta_expr
  gamma = reduced.gamma_expr

  dalpha = reduced.dalpha_expr
  dbeta = ca.jacobian(beta, arg)
  ddelta = ca.jacobian(delta, arg)
  dgamma = reduced.dgamma_expr

  beta_ = ca.substitute(beta, arg, x_sing)
  delta_ = ca.substitute(delta, arg, x_sing)

  dalpha_ = ca.substitute(dalpha, arg, x_sing)
  dbeta_ = ca.substitute(dbeta, arg, x_sing)
  ddelta_ = ca.substitute(ddelta, arg, x_sing)
  dgamma_ = ca.substitute(dgamma, arg, x_sing)

  num = dbeta_ * dx_sing**3 + ddelta_ * dx_sing**2 + dgamma_ * dx_sing
  den = dalpha_ * dx_sing + 2 * beta_ * dx_sing + delta_
  return -float(ca.evalf(num / den))

def plot_phase():
  par = ball_and_beam_parameters_default
  par.ball_airdrag_coef = 0.5
  dynamics = BallAndBeamDynamics(par, auto_compute=False)

  lam = ca.SX.sym('lam')
  lam_shift = -0.0580935
  Q_expr = ca.vertcat(
    0.3 * ca.sin(lam - lam_shift),
    1.0 * ca.sin(lam + lam_shift)
  )
  Q = ca.Function('Q', [lam], [Q_expr])

  reduced = ReducedDynamics(dynamics, Q)
  sings = get_singularities(reduced)
  equils = get_equilibriums(reduced)

  v_max = 4

  def rhs(nu, state):
    lam, v = state
    dlam = reduced.alpha(lam) * v
    dv = -reduced.beta(lam) * v**2 - reduced.delta(lam) * v - reduced.gamma(lam)
    return [float(dlam), float(dv)]

  def stop_cond(nu, state):
    lam, v = state
    if abs(v) > v_max or lam < -np.pi or lam > 3 * np.pi:
      return -1
    return 1

  stop_cond.terminal = True

  eps = 0.01

  sing1 = sings[0]
  lam0, v1, v0 = sing1
  a0 = get_accel_at_sing(lam0, v0, reduced)

  plt.figure('phase portrait')

  sol = solve_ivp(rhs, [0, -200], [lam0 + v0 * eps, v0 + a0 * eps], max_step=1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color='#6060A0')

  sol = solve_ivp(rhs, [0, -200], [lam0 - v0 * eps, v0 - a0 * eps], max_step=1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color='#6060A0')

  a0 = get_accel_at_sing(lam0, v1, reduced)
  sol = solve_ivp(rhs, [0, 200], [lam0 + v1 * eps, v1 + a0 * eps], max_step=1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color="#E57129")

  sol = solve_ivp(rhs, [0, 200], [lam0 - v1 * eps, v1 - a0 * eps], max_step=1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color='#E57129')

  sing2 = sings[1]
  lam0, v1, v0 = sing2
  a0 = get_accel_at_sing(lam0, v0, reduced)

  sol = solve_ivp(rhs, [0, 200], [lam0 - v0 * eps, v0 - a0 * eps], max_step=1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color='#6060A0')

  sol = solve_ivp(rhs, [0, 200], [lam0 + v0 * eps, v0 + a0 * eps], max_step=1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color='#6060A0')

  a0 = get_accel_at_sing(lam0, v1, reduced)
  sol = solve_ivp(rhs, [0, -200], [lam0 + v1 * eps, v1 + a0 * eps], max_step=1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color="#E57129")

  sol = solve_ivp(rhs, [0, -200], [lam0 - v1 * eps, v1 - a0 * eps], max_step=1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color='#E57129')

  lam0 = 0
  for v0 in np.linspace(0.01, v_max, 10):
    sol = solve_ivp(rhs, [0, 100], [lam0, v0], max_step=1, events=stop_cond)
    plt.plot(sol.y[0], sol.y[1], color='#A0A0A0', lw=1, alpha=1)
    plt.plot(sol.y[0] + 2 * np.pi, sol.y[1], color='#A0A0A0', lw=1, alpha=1)

    sol = solve_ivp(rhs, [0, -100], [lam0, v0], max_step=1, events=stop_cond)
    plt.plot(sol.y[0], sol.y[1], color='#A0A0A0', lw=1, alpha=1)
    plt.plot(sol.y[0] + 2 * np.pi, sol.y[1], color='#A0A0A0', lw=1, alpha=1)

    sol = solve_ivp(rhs, [0, -100], [lam0, -v0], max_step=1, events=stop_cond)
    plt.plot(sol.y[0], sol.y[1], color='#A0A0A0', lw=1, alpha=1)
    plt.plot(sol.y[0] + 2 * np.pi, sol.y[1], color='#A0A0A0', lw=1, alpha=1)

    sol = solve_ivp(rhs, [0, 100], [lam0, -v0], max_step=1, events=stop_cond)
    plt.plot(sol.y[0], sol.y[1], color='#A0A0A0', lw=1, alpha=1)
    plt.plot(sol.y[0] + 2 * np.pi, sol.y[1], color='#A0A0A0', lw=1, alpha=1)

  lam0 = np.pi
  for v0 in np.linspace(0.01, v_max, 10):
    sol = solve_ivp(rhs, [0, 100], [lam0, v0], max_step=1, events=stop_cond)
    plt.plot(sol.y[0], sol.y[1], color='#A0A0A0', lw=1, alpha=1)

    sol = solve_ivp(rhs, [0, -100], [lam0, v0], max_step=1, events=stop_cond)
    plt.plot(sol.y[0], sol.y[1], color='#A0A0A0', lw=1, alpha=1)

    sol = solve_ivp(rhs, [0, 100], [lam0, -v0], max_step=1, events=stop_cond)
    plt.plot(sol.y[0], sol.y[1], color='#A0A0A0', lw=1, alpha=1)

    sol = solve_ivp(rhs, [0, -100], [lam0, -v0], max_step=1, events=stop_cond)
    plt.plot(sol.y[0], sol.y[1], color='#A0A0A0', lw=1, alpha=1)

  # limit cycle
  sol = solve_ivp(rhs, [0, -300], [np.pi, 1.14141], max_step=0.1, events=stop_cond)
  plt.plot(sol.y[0], sol.y[1], color='#A0F0A0', lw=2, alpha=1)

  plt.xlim(0, 2*np.pi)
  plt.ylim(-v_max, v_max)
  for sing in sings:
    plt.axvline(sing[0], color='red', lw=1, ls='--')
    plt.plot(sing[0], sing[1], 'x', color='red')
    plt.plot(sing[0], sing[2], 'x', color='red')

  for eq in equils:
    plt.plot(eq, 0, '.', color='blue')

  set_pi_xticks('1/2')
  plt.grid(True, zorder=-1)
  plt.tight_layout(pad=0.1)

def plot_coefs():
  lam = ca.SX.sym('lam')
  eps = 0.1
  Q = ca.vertcat(
    ca.sin(lam - eps),
    ca.sin(lam)
  )
  dQ = ca.jacobian(Q, lam)
  ddQ = ca.jacobian(dQ, lam)

  par = ball_and_beam_parameters_default
  par.ball_airdrag_coef = 0.1
  dynamics = BallAndBeamDynamics(par, auto_compute=False)
  alpha_expr = dynamics.B_perp(Q) @ dynamics.M(Q) @ dQ
  beta_expr = dynamics.B_perp(Q) @ dynamics.M(Q) @ ddQ + \
    dynamics.B_perp(Q) @ dynamics.C(Q, dQ) @ dQ
  gamma_expr = dynamics.B_perp(Q) @ dynamics.G(Q)
  delta_expr = dynamics.B_perp(Q) @ dynamics.D(Q) @ dQ

  alpha = ca.Function('alpha', [lam], [alpha_expr])
  beta = ca.Function('beta', [lam], [beta_expr])
  gamma = ca.Function('gamma', [lam], [gamma_expr])
  delta = ca.Function('delta', [lam], [delta_expr])

  lam_arr = np.linspace(0, 2 * np.pi, 200)
  alpha_arr = map_array(alpha, lam_arr, elem_size=1)
  beta_arr = map_array(beta, lam_arr, elem_size=1)
  gamma_arr = map_array(gamma, lam_arr, elem_size=1)
  delta_arr = map_array(delta, lam_arr, elem_size=1)

  sings = find_all_roots(alpha, [lam_arr[0], lam_arr[-1]])

  fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7, 8), num='coefs of alpha beta delta gamma')
  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(lam_arr, alpha_arr)
  plt.axhline(0)
  plt.ylabel('alpha')

  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(lam_arr, beta_arr)
  plt.ylabel('beta')

  plt.sca(axes[2])
  plt.grid(True)
  plt.plot(lam_arr, gamma_arr)
  plt.ylabel('gamma')

  plt.sca(axes[3])
  plt.grid(True)
  plt.plot(lam_arr, delta_arr)
  plt.ylabel('delta')

  for sing in sings:
    for ax in axes:
      ax.axvline(sing, color='red', lw=1, ls='--')

  plt.tight_layout(h_pad=0.1)
  set_pi_xticks('1/4')


if __name__ == '__main__':
  # plot_coefs()
  plot_phase()
  plt.show()