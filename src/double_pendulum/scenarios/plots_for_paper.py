from common.mechsys import MechanicalSystem
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default,
  convert_parameters
)
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from double_pendulum.scenarios.singular_oscillations_planner import (
  show_trajectory_projections,
  show_reduced_dynamics_phase_prortrait,
)
from double_pendulum.anim import (
  motion_schematic,
  motion_schematic_v2
)
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from singular_motion_planner.singular_constrs import get_sing_constr_at


def make_constr(par : DoublePendulumParam, q_sing : np.ndarray, scale=1):
  B = ca.DM([[1], [0]])
  B_perp = ca.DM([[0, -1]])

  par2 = convert_parameters(par)
  p2 = par2.p2
  p3 = par2.p3
  c1 = ca.DM([[p3], [-p2 * ca.cos(q_sing[1]) - p3]])
  k = ca.sqrt(2) / 2 * p2 * p3 - p2**2 / 3
  c2 = B_perp.T

  print('VHC k: ', k)
  print('VHC c1: ', c1)
  print('VHC c2: ', c2)

  phi = ca.SX.sym('phi')
  constr_expr = q_sing + c1 * phi + c2 * k * phi**2 / 2
  constr_fun = ca.Function('constr', [phi], [constr_expr])
  return constr_fun

def find_trajectory():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  constr = make_constr(par, [-np.pi/4, 3*np.pi/4])
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-2, -0.5e-3], 0.0, max_step=1e-3)
  tr_right = solve_reduced(reduced, [2., 0.5e-3], 0.0, max_step=1e-3)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_closed)

  show_trajectory_projections(tr_orig)
  show_reduced_dynamics_phase_prortrait(reduced, tr_closed)
  motion_schematic(tr_orig, par)
  plt.show()

if __name__ == '__main__':
  find_trajectory()
