import numpy as np
from scipy.integrate import solve_ivp
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
)
from common.trajectory import Trajectory
import casadi as ca
import matplotlib.pyplot as plt
from double_pendulum.anim import draw
import json
from double_pendulum.motion_planner.reduced_dynamics import ReducedDynamics, solve_reduced


def get_test_par() -> DoublePendulumParam:
  return DoublePendulumParam(
    lengths=[1., 1.],
    mass_centers=[0.5, 0.5],
    masses=[0.2, 0.2],
    inertia=[0.05, 0.05],
    actiated_joint=0,
    gravity_accel=9.81
  )

def test7():
  par = get_test_par()

  I1,I2 = par.inertia
  c1,c2 = par.mass_centers
  m1,m2 = par.masses
  l1,l2 = par.lengths
  g = par.gravity_accel

  p1 = I1 + I2 + c1**2 * m1 + c2**2 * m2 + l1**2 * m2
  p2 = m2 * c2 * l1
  p3 = I2 + m2 * c2**2
  p4 = c1 * m1 + l1 * m2
  p5 = c2 * m2

  s_left = -0.04
  s_right = -s_left
  # qs1, qs2 = -1., 2.4
  # qs1, qs2 = -1., 2.7
  # qs1, qs2 = -0.5, 2.5
  # qs1, qs2 = -1., 1 + np.pi/2
  # qs1, qs2 = 3*np.pi/4, np.pi/2 - 3*np.pi/4
  # qs1, qs2 = np.pi/2 + np.pi/3, np.pi/2 - np.pi/3 - np.pi/2
  # qs1, qs2 = 2*np.pi/3, np.pi/2 - 2*np.pi/3
  qs1, qs2 = -np.pi/3, np.pi/2 + np.pi/3
  # qs1, qs2 = -2., 2.8
  # qs1, qs2 = 2.5, 1.3
  # qs1, qs2 = 2.1, 1.3
  # qs1, qs2 = -1, 0.6
  # qs1, qs2 = -1.4, 0.7
  # qs1, qs2 = -2.5, 2.8
  qs = ca.DM([qs1, qs2])

  # q1 + q2 = pi/2
  # q2 in [-pi/2, 0] U [pi/2, pi] 

  dynamics = DoublePendulumDynamics(par)
  k = ca.SX.sym('k')
  B_perp = ca.DM([[0, 1]])
  N = ca.pinv(dynamics.M_expr) @ dynamics.B_expr
  L = dynamics.M_expr @ B_perp.T

  dalpha = ca.substitute(
    k * L.T @ L + ca.jtimes(L, dynamics.q, N).T @ N,
    dynamics.q,
    qs
  )
  beta = ca.substitute(
    k * L.T @ L + B_perp @ dynamics.C(qs, N) @ N,
    dynamics.q,
    qs
  )
  gamma = ca.substitute(
    B_perp @ dynamics.G_expr,
    dynamics.q,
    qs
  )
  ineq1 = -beta / dalpha - 0.5 - 0.5
  ineq2 = gamma / dalpha - 0.1
  ineq3 = [k, -k]

  # nlp
  decision_variables = k
  constraints = ca.vertcat(ineq1, ineq2, *ineq3)
  constraints_lb = ca.vertcat(0.0, 0.0, -100, -100)
  cost_function = 0

  nlp = {
      'x': decision_variables,
      'f': cost_function,
      'g': constraints
  }

  while True:
    dv0 = 200 * np.random.rand(*decision_variables.shape) - 100
    dv0 = ca.DM(dv0)
    solver = ca.nlpsol('BVP', 'ipopt', nlp)
    sol = solver(x0=dv0, lbg=constraints_lb)

    ok = np.all(np.array(ca.DM(ca.substitute(constraints, decision_variables, sol['x']) >= constraints_lb), bool))
    if not ok:
      continue

    k_star = float(ca.substitute(k, decision_variables, sol['x']))
    N_star = ca.substitute(N, dynamics.q, qs)
    L_star = ca.substitute(L, dynamics.q, qs)
    s = ca.SX.sym('s')
    constr_expr = qs + N_star * s + k_star * L_star * s**2 / 2
    constr = ca.Function('constr', [s], [constr_expr])
    reduced = ReducedDynamics(dynamics, constr)

    ok = reduced.alpha(s_left) * reduced.alpha(s_right) < 0
    if not ok:
      continue

    ok = reduced.gamma(s_left) * reduced.gamma(s_right) > 0
    if not ok:
      continue

    break

  coefs = np.reshape([qs, ca.DM(N_star), ca.DM(k_star * L_star)], (3, 2))
  print('poly:', np.array2string(coefs, separator=', '))

  cfg = {
    'qs': [float(qs[0]), float(qs[1])],
    'N': [float(N_star[0]), float(N_star[1])],
    'L': [float(L_star[0]), float(L_star[1])],
  }
  with open('data/cfg.json', 'w') as f:
    json.dump(cfg, f)

  with open('data/parameters.json', 'w') as f:
    json.dump(par.todict(), f)

  _,axes = plt.subplots(3, 1, sharex=True)
  s = np.linspace(s_left, s_right)
  alpha = [float(reduced.alpha(e)) for e in s]
  beta = [float(reduced.beta(e)) for e in s]
  gamma = [float(reduced.gamma(e)) for e in s]
  plt.sca(axes[0])
  plt.grid(True)
  plt.axhline(0, color='black', lw=1)
  plt.plot(s, alpha)
  plt.sca(axes[1])
  plt.grid(True)
  plt.axhline(0, color='black', lw=1)
  plt.plot(s, beta)
  plt.sca(axes[2])
  plt.grid(True)
  plt.axhline(0, color='black', lw=1)
  plt.plot(s, gamma)
  plt.tight_layout()
  plt.savefig('data/alpha-beta-gamma.pdf')

  plt.figure('phase')
  plt.axhline(0, color='black', lw=1)
  plt.axvline(0, color='black', lw=1)
  plt.grid(True)
  tr1 = solve_reduced(reduced, [s_left, -0.0001], 0.0, max_step=1e-4)
  plt.plot(tr1.coords, tr1.vels, color='lightblue', lw=2)
  plt.plot(tr1.coords, -tr1.vels, color='lightblue', lw=2)
  tr2 = solve_reduced(reduced, [s_right, 0.0001], 0.0, max_step=1e-4)
  plt.plot(tr2.coords, tr2.vels, color='lightblue', lw=2)
  plt.plot(tr2.coords, -tr2.vels, color='lightblue', lw=2)
  plt.savefig('data/phase.pdf')

  plt.figure()
  plt.grid(True)
  draw([qs1, qs2], par)
  plt.savefig('data/configuration.pdf')

  plt.show()

def test8():
  pass

if __name__ == '__main__':
  np.set_printoptions(suppress=True)
  test7()
