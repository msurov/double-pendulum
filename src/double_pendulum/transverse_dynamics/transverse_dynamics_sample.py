from double_pendulum.motion_planner.singular_constrs import get_sing_constr_at
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default
)
import numpy as np
from double_pendulum.motion_planner.reduced_dynamics import (
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
import matplotlib.pyplot as plt
from double_pendulum.transverse_dynamics.transverse_dynamics import (
  TransverseCoordinates,
  TransverseCoordinatesPar,
  TransverseDynamics
)

def sample_transverse_dynamics():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  q_sing = np.array([-2, 0.9])
  constr = get_sing_constr_at(dynamics, q_sing)
  reduced = ReducedDynamics(dynamics, constr)

  tr_left = solve_reduced(reduced, [-0.02, -1e-4], 0.0, max_step=1e-4)
  tr_right = solve_reduced(reduced, [0.02, 1e-4], 0.0, max_step=1e-4)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_closed)
  trans_par = TransverseCoordinatesPar(
    transverse_projection_mat=np.array([
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]),
    proj_plane_x = np.array([100., 0., 0., 0.]),
    proj_plane_y = np.array([0., 0., 1., 0.]),
    proj_plane_origin=np.concatenate((q_sing, [0, 0]))
  )
  coords = TransverseCoordinates(tr_orig, trans_par)

sample_transverse_dynamics()
