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
from common.numpy_utils import (
  cont_angle,
  map_array
)
from transverse_dynamics.cylindrical_transverse_coordinates import (
  CylindricalTransverseCoordinates,
  CylindricalTransverseCoordinatesPar,
)
from transverse_dynamics.transverse_dynamics import (
  TransverseDynamics
)
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np


def make_sample_data():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  singpt = np.array([-np.pi/4, 3*np.pi/4])
  constr = get_sing_constr_at(dynamics, singpt, 100)
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-5., -1e-3], 0.0, max_step=1e-3)
  tr_right = solve_reduced(reduced, [5., 1e-3], 0.0, max_step=1e-3)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_closed)

  trans_par = CylindricalTransverseCoordinatesPar(
    transverse_projection_mat = np.array([
      [1/0.4, 1/0.125, 0.,  0.],
      [ 0.,  0., 1/10., 1/3.333],
    ]),
    proj_plane_x = np.array([1/0.4, -1/0.125, 0.,  0.]),
    proj_plane_y = np.array([ 0.,   0., -1/10., 1/3.333]),
    proj_plane_origin = np.concatenate((singpt, [0, 0]))
  )

  coords = CylindricalTransverseCoordinates(tr_orig, trans_par)
  traj_of_time = make_interp_spline(tr_orig.time, tr_orig.phase, k=5, bc_type='periodic')
  ctrl_of_time = make_interp_spline(tr_orig.time, tr_orig.control, k=3, bc_type='periodic')
  trans_dyn = TransverseDynamics(dynamics, coords)
  theta = map_array(lambda x: coords.forward_transform_fun(x)[0], tr_orig.phase, 1)
  cont_angle(theta)
  traj_of_theta = make_interp_spline(theta, tr_orig.phase, k=5, bc_type='periodic')
  ctrl_of_theta = make_interp_spline(theta, tr_orig.control, k=3, bc_type='periodic')

  return {
    'dynamics_par': par,
    'dynamics': dynamics,
    'trans_par': trans_par,
    'trans_dyn': trans_dyn,
    'constr': constr,
    'reduced': reduced,
    'traj_reduced': tr_closed,
    'coords': coords,
    'traj': tr_orig,
    'traj_of_time': traj_of_time,
    'control_of_time': ctrl_of_time,
    'traj_of_theta': traj_of_theta,
    'control_of_theta': ctrl_of_theta,
    'traj_period': tr_orig.time[-1]
  }
