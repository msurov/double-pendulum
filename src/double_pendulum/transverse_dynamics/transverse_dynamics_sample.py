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
  TransverseDynamics,
  compute_theta,
  compute_transverse
)
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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

def verify_free_motion_transverse_dynamics():
  sampledata = make_sample_data()
  coords = sampledata['coords']
  transdyn = sampledata['trans_dyn']
  dynamics = sampledata['dynamics']

  plt.title('Comparision of trajectories of the original\n dynamics and the transverse dynamics with zero input')

  def rhs_trans(theta, xi):
    dxi = transdyn.D_xi_fun(theta, xi, 0.)
    return np.reshape(dxi, (-1,))

  np.random.seed(0)
  theta0 = 2.
  xi0 = 0.1 * np.random.normal(size=3)
  sol = solve_ivp(rhs_trans, [theta0, theta0 + 2.], xi0, max_step=1e-3)
  xi = sol.y.T
  theta = sol.t
  lines_transverse = plt.plot(theta, xi)

  par = sampledata['par']
  dyn = sampledata['dynamics']
  x0 = coords.inverse_transform_fun(theta0, xi0)
  x0 = np.reshape(x0, (4,))

  def rhs_orig(t, x):
    theta,_ = compute_transverse(x, coords)
    u_ref = coords.usp(theta)
    dx = dyn.rhs(x, u_ref)
    return np.reshape(dx, (-1,))

  sol = solve_ivp(rhs_orig, [0., 0.06], x0, max_step=1e-4)
  theta, xi = compute_transverse(sol.y.T, coords)
  lines_orig = plt.plot(theta, xi, ls='--', lw=2)

  plt.legend(lines_transverse + lines_orig, 
    [R'$\xi_{1,tran}$', R'$\xi_{2,tran}$', R'$\xi_{3,tran}$'] + 
    [R'$\xi_{1,orig}$', R'$\xi_{2,orig}$', R'$\xi_{3,orig}$'])

  plt.xlabel(R'$\theta$')
  plt.ylabel(R'$\xi$')
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def verify_transverse_dynamics():
  sampledata = make_sample_data()
  coords = sampledata['coords']
  transdyn = sampledata['trans_dyn']
  dynamics = sampledata['dynamics']

  plt.title('Comparision of trajectories of the original\n dynamics and the transverse dynamics with nonzero input')

  def stab_input(theta):
    return 2. * np.sin(theta)

  def rhs_trans(theta, xi):
    dxi = transdyn.D_xi_fun(theta, xi, stab_input(theta))
    return np.reshape(dxi, (-1,))

  np.random.seed(0)
  theta0 = 1.
  xi0 = 0.1 * np.random.normal(size=3)
  sol = solve_ivp(rhs_trans, [theta0, theta0 + 2.5], xi0, max_step=1e-3)
  xi = sol.y.T
  theta = sol.t
  lines_transverse = plt.plot(theta, xi)

  par = sampledata['par']
  dyn = sampledata['dynamics']
  x0 = coords.inverse_transform_fun(theta0, xi0)
  x0 = np.reshape(x0, (4,))

  def rhs_orig(t, x):
    theta = compute_theta(x, par)
    u = coords.usp(theta) + stab_input(theta)
    dx = dyn.rhs(x, u)
    return np.reshape(dx, (-1,))

  sol = solve_ivp(rhs_orig, [0., 0.06], x0, max_step=1e-4)
  theta, xi = compute_transverse(sol.y.T, coords)
  lines_orig = plt.plot(theta, xi, ls='--', lw=2)

  plt.legend(lines_transverse + lines_orig, 
    [R'$\xi_{1,tran}$', R'$\xi_{2,tran}$', R'$\xi_{3,tran}$'] + 
    [R'$\xi_{1,orig}$', R'$\xi_{2,orig}$', R'$\xi_{3,orig}$'])

  plt.xlabel(R'$\theta$')
  plt.ylabel(R'$\xi$')
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def test_transverse_linearization():
  sampledata = make_sample_data()
  coords = sampledata['coords']
  transdyn = sampledata['trans_dyn']
  dynamics = sampledata['dynamics']

  plt.title('Comparision of trajectories of \n the transverse dynamics and its linearization')

  def stab_input(theta):
    return 5. * np.sin(theta)

  def rhs_trans_lin(theta, xi):
    dxi = transdyn.A_fun(theta) @ xi + transdyn.B_fun(theta) * stab_input(theta)
    return np.reshape(dxi, (-1,))

  np.random.seed(0)
  theta0 = 1.
  xi0 = 0.1 * np.random.normal(size=3)
  sol = solve_ivp(rhs_trans_lin, [theta0, theta0 + 1.], xi0, max_step=1e-3)
  xi = sol.y.T
  theta = sol.t
  lines_linearized = plt.plot(theta, xi)

  def rhs_trans(theta, xi):
    dxi = transdyn.D_xi_fun(theta, xi, stab_input(theta))
    return np.reshape(dxi, (-1,))

  sol = solve_ivp(rhs_trans, [theta0, theta0 + 1.], xi0, max_step=1e-3)
  xi = sol.y.T
  theta = sol.t
  lines_transverse = plt.plot(theta, xi, lw=2, ls='--')

  plt.legend(lines_linearized + lines_transverse, 
    [R'$\xi_{1,lin}$', R'$\xi_{2,lin}$', R'$\xi_{3,lin}$'] + 
    [R'$\xi_{1,nonlin}$', R'$\xi_{2,nonlin}$', R'$\xi_{3,nonlin}$'])
  plt.grid(True)
  plt.show()

def main():
  verify_free_motion_transverse_dynamics()
  verify_transverse_dynamics()
  test_transverse_linearization()

if __name__ == '__main__':
  main()
