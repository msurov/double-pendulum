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


@dataclass
class CollocationPar:
  nbasis : int
  atol : float = 1e-5
  nknots : Optional[int] = None

  def __post_init__(self):
    if self.nknots is None:
      self.nknots = 2 * self.nbasis

def find_periodic_solution(dynamics : MechanicalSystem, 
                       cyclic_vhc_expr : ca.SX, 
                       vhc_polar_arg : ca.SX,
                       vhc_par : ca.SX,
                       par_diap : Tuple[float, float],
                       par : CollocationPar
                      ):

  Q = cyclic_vhc_expr
  dQ = ca.jacobian(Q, vhc_polar_arg)
  ddQ = ca.jacobian(dQ, vhc_polar_arg)
  B_perp = dynamics.B_perp(Q)
  M = dynamics.M(Q)
  C = dynamics.C(Q, dQ)
  D = dynamics.D(Q)
  G = dynamics.G(Q)

  alpha_expr = B_perp @ M @ dQ
  dalpha_expr = ca.jacobian(alpha_expr, vhc_polar_arg)
  beta_expr = B_perp @ (M @ ddQ + C @ dQ)
  gamma_expr = B_perp @ G
  delta_expr = B_perp @ D @ dQ

  nbasis = par.nbasis
  assert nbasis > 2
  assert nbasis % 2 == 1

  c = ca.SX.sym('c', nbasis)
  v = c[0]

  for k in np.arange(1, nbasis // 2 + 1):
    v += c[2 * k - 1] * ca.sin(k * vhc_polar_arg)
    v += c[2 * k] * ca.cos(k * vhc_polar_arg)

  dv = ca.jacobian(v, vhc_polar_arg)

  equation = alpha_expr * v * dv + beta_expr * v**2 + delta_expr * v + gamma_expr
  equation2 = ca.jacobian(equation, vhc_polar_arg)

  nknots = par.nknots
  knots = 2 * np.pi * (np.arange(0, nknots) + 0.5) / nknots

  constraints = []
  for knot in knots:
    constr = ca.substitute(equation, vhc_polar_arg, knot)
    constraints.append(constr)
    constr = ca.substitute(equation2, vhc_polar_arg, knot)
    constraints.append(constr)

  constraints = ca.vertcat(*constraints)
  dec_var = ca.vertcat(c, vhc_par)
  dec_var_lower = ca.vertcat(-10. * ca.DM.ones(c.shape), par_diap[0])
  dec_var_upper = ca.vertcat(10. * ca.DM.ones(c.shape), par_diap[1])

  nlp = {
    'x': dec_var,
    'g': constraints,
    'f': 1
  }
  solver = ca.nlpsol('solver', 'ipopt', nlp)
  initial_guess = np.zeros(dec_var.shape)
  initial_guess[0] = 1

  initial_guess = ca.substitute(dec_var, c[0], 1)
  initial_guess = ca.substitute(initial_guess, c, 0)
  initial_guess = ca.substitute(initial_guess, vhc_par, (par_diap[0] + par_diap[1]) / 2)
  initial_guess = ca.evalf(initial_guess)

  solution = solver(x0 = initial_guess, 
                      lbg = -par.atol,
                      ubg = par.atol, 
                      lbx = dec_var_lower,
                      ubx = dec_var_upper
                    )
  ok = np.all(np.abs(solution['g'].full().flatten()) <= par.atol)
  if not ok:
    return None

  v_res = ca.substitute(v, dec_var, solution['x'])
  v_fun = ca.Function('v', [vhc_polar_arg], [v_res])
  par_value = ca.substitute(vhc_par, dec_var, solution['x'])
  return v_fun, par_value

def compute_periodic_trajectory(dynamics : BallAndBeamDynamics):
  lam = ca.SX.sym('lam')
  lam_shift = ca.SX.sym('lam_shift')
  Q_expr = ca.vertcat(
    0.1 * ca.sin(lam - lam_shift),
    0.4 * ca.sin(lam + lam_shift)
  )

  par = CollocationPar(nbasis=25, nknots=55, atol=1e-7)
  sol, lam_shift_ = find_periodic_solution(dynamics, Q_expr, lam, lam_shift, [-1., 1.], par)
  lam_ = np.linspace(0, 2 * np.pi, 300)
  dlam_ = map_array(sol, lam_, 1)
  reduced_traj = make_traj(lam_, dlam_)

  Q_impl = ca.substitute(Q_expr, lam_shift, lam_shift_)
  Q = ca.Function('Q', [lam], [Q_impl])
  reduced = ReducedDynamics(dynamics, Q)

  traj = reconstruct_trajectory(Q, reduced, dynamics, reduced_traj)
  return traj
