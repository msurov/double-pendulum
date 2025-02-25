from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  double_pendulum_param_default
)
from singular_motion_planner.singular_constrs import get_sing_constr_at
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  reconstruct_trajectory
)
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from transverse_dynamics.transverse_coordinates import (
  TransverseCoordinates,
  TransverseCoordinatesPar,
  TransverseDynamics,
)
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np


def make_sample_data():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  singpt = np.array([-2.2, 1.12])
  constr = get_sing_constr_at(dynamics, singpt)
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-0.05, -1e-4], 0.0, max_step=1e-4)
  tr_right = solve_reduced(reduced, [0.08, 1e-4], 0.0, max_step=1e-4)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_closed)

  trans_par = TransverseCoordinatesPar(
    transverse_projection_mat = np.array([
      [20., 20., 0., 0.],
      [ 0.,  0., 1., 1.],
    ]),
    proj_plane_x = np.array([20., -20., 0.,  0.]),
    proj_plane_y = np.array([ 0.,   0., 1., -1.]),
    proj_plane_origin = np.concatenate((singpt, [0, 0]))
  )

  coords = TransverseCoordinates(tr_orig, trans_par)
  trajsp = make_interp_spline(tr_orig.time, tr_orig.phase, k=5, bc_type='periodic')
  trans_dyn = TransverseDynamics(dynamics, coords)

  return {
    'dynamics_par': par,
    'dynamics': dynamics,
    'trans_par': trans_par,
    'trans_dyn': trans_dyn,
    'constr': constr,
    'reduced': reduced,
    'coords': coords,
    'traj': tr_orig,
    'traj_spline': trajsp,
    'traj_period': tr_orig.time[-1]
  }
