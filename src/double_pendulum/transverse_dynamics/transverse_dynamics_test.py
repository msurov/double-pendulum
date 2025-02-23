import numpy as np
from double_pendulum.motion_planner.singular_constrs import get_sing_constr_at
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default
)
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
from common.numpy_utils import are_angles_close
from double_pendulum.transverse_dynamics.transverse_dynamics import (
  TransverseCoordinates,
  TransverseCoordinatesPar,
  TransverseDynamics
)
from scipy.interpolate import make_interp_spline


def make_sample_data():
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
      [0., 100., 0., 0.],
      [0., 0., 0., 1.],
    ]),
    proj_plane_x = np.array([100., 0., 0., 0.]),
    proj_plane_y = np.array([0., 0., 1., 0.]),
    proj_plane_origin=np.concatenate((q_sing, [0, 0]))
  )
  coords = TransverseCoordinates(tr_orig, trans_par)
  trajsp = make_interp_spline(tr_orig.time, tr_orig.phase, k=5, bc_type='periodic')
  return {
    'par': trans_par,
    'dynamics': dynamics,
    'constr': constr,
    'reduced': reduced,
    'coords': coords,
    'traj': tr_orig,
    'traj_spline': trajsp,
    'traj_period': tr_orig.time[-1]
  }

def test_forward_transform():
  sampledata = make_sample_data()
  traj = sampledata['traj']
  coords = sampledata['coords']

  for i in range(traj.time.shape[0]):
    x = traj.phase[i]
    val = coords.forward_transform_fun(x)
    xi = val[1:]
    assert np.allclose(xi, 0)
  
  maxtime = 2 * sampledata['traj_period']
  trajsp = sampledata['traj_spline']

  np.random.seed(0)
  for i in range(100):
    t = maxtime * np.random.rand()
    x = trajsp(t)
    val = coords.forward_transform_fun(x)
    xi = val[1:]
    assert np.allclose(xi, 0, atol=1e-5)

def test_inv_transform():
  sampledata = make_sample_data()
  traj = sampledata['traj']
  coords = sampledata['coords']
  
  np.random.seed(0)
  for i in range(100):
    xi1 = 1e-2 * np.random.normal(size=3)
    theta1 = 4 * np.pi * np.random.rand()
    x = coords.inverse_transform_fun(theta1, xi1)
    val = coords.forward_transform_fun(x)
    theta2 = float(val[0])
    xi2 = np.reshape(val[1:], (3,))
    assert np.allclose(xi1, xi2)
    assert are_angles_close(theta1, theta2)

def test_transverse_dynamics():
  sampledata = make_sample_data()

def test_transverse_linearization():
  pass
