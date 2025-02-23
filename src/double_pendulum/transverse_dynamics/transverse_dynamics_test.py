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
from common.numpy_utils import (
  are_angles_close,
  normalize_angle,
  vectors_dif,
  cont_angle
)
from double_pendulum.transverse_dynamics.transverse_dynamics import (
  TransverseCoordinates,
  TransverseCoordinatesPar,
  TransverseDynamics
)
from scipy.interpolate import make_interp_spline
import casadi as ca


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
    transverse_projection_mat=np.array([
      [100., 100., 0., 0.],
      [0., 0., 1., 1.],
    ]),
    proj_plane_x = np.array([100., -100., 0., 0.]),
    proj_plane_y = np.array([0., 0., 1., -1.]),
    proj_plane_origin=np.concatenate((singpt, [0, 0]))
  )

  coords = TransverseCoordinates(tr_orig, trans_par)
  trajsp = make_interp_spline(tr_orig.time, tr_orig.phase, k=5, bc_type='periodic')
  trans_dyn = TransverseDynamics(dynamics, coords)

  return {
    'par': trans_par,
    'dynamics': dynamics,
    'trans_dyn': trans_dyn,
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
    assert np.allclose(xi, 0, atol=1e-4)

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

def test_ref_control():
  pass

def test_inverse_jacobian():
  sampledata = make_sample_data()
  traj = sampledata['traj']
  coords = sampledata['coords']

  np.random.seed(0)

  for i in range(100):
    xi1 = 5e-2 * np.random.normal(size=3)
    theta1 = 2 * np.pi * np.random.rand()
    x1 = coords.inverse_transform_fun(theta1, xi1)

    xi2 = xi1 + 1e-4 * np.random.normal(size=3)
    theta2 = theta1 + 0.01 * np.random.rand()
    x2 = coords.inverse_transform_fun(theta2, xi2)
    J = coords.inverse_jac_fun(theta1, xi1)

    dif1 = x2 - x1
    dif2 = J @ ca.vertcat(theta2 - theta1, xi2 - xi1)

    assert vectors_dif(dif1, dif2) < 1e-2

def test_forward_jacobian():
  sampledata = make_sample_data()
  traj_spline = sampledata['traj_spline']
  coords = sampledata['coords']
  max_time = 2 * sampledata['traj_period']

  np.random.seed(0)

  for i in range(100):
    t = max_time * np.random.rand()
    x_ref = traj_spline(t)

    x1 = x_ref * (1 + 1e-3 * np.random.normal(size=x_ref.shape))
    x2 = x_ref * (1 + 1e-3 * np.random.normal(size=x_ref.shape))

    val1 = coords.forward_transform_fun(x1)
    val2 = coords.forward_transform_fun(x2)

    J = coords.forward_jac_fun(x1)
    dif1 = val2 - val1
    dif2 = J @ (x2 - x1)

    assert vectors_dif(dif1, dif2) < 1e-2

def test_jacobians():
  sampledata = make_sample_data()
  traj = sampledata['traj']
  coords = sampledata['coords']

  np.random.seed(0)

  for i in range(100):
    xi = 1e-2 * np.random.normal(size=3)
    theta = 4 * np.pi * np.random.rand()
    x = coords.inverse_transform_fun(theta, xi)
    J = coords.forward_jac_fun(x)
    Jinv = coords.inverse_jac_fun(theta, xi)
    assert np.allclose(J @ Jinv, np.eye(4))


test_forward_transform()
test_inv_transform()
test_jacobians()
test_forward_jacobian()
test_inverse_jacobian()

