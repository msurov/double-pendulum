import numpy as np
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from common.plots import set_pi_xticks, set_pi_yticks
import matplotlib.pyplot as plt
from transverse_dynamics.transverse_coordinates import (
  compute_theta,
  compute_transverse
)
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import casadi as ca
from double_pendulum.scenarios.sample_data import make_sample_data


def verify_free_motion_transverse_dynamics():
  sampledata = make_sample_data()
  coords = sampledata['coords']
  transdyn = sampledata['trans_dyn']
  dynamics = sampledata['dynamics']

  plt.figure('verify_free_motion_transverse_dynamics')
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

def verify_transverse_dynamics():
  sampledata = make_sample_data()
  coords = sampledata['coords']
  transdyn = sampledata['trans_dyn']
  dynamics = sampledata['dynamics']

  plt.figure('verify_transverse_dynamics')
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

  trans_par = sampledata['trans_par']
  dyn = sampledata['dynamics']
  x0 = coords.inverse_transform_fun(theta0, xi0)
  x0 = np.reshape(x0, (4,))

  def rhs_orig(t, x):
    theta = compute_theta(x, trans_par)
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

def test_transverse_linearization():
  sampledata = make_sample_data()
  coords = sampledata['coords']
  transdyn = sampledata['trans_dyn']
  dynamics = sampledata['dynamics']

  plt.figure('test_transverse_linearization')
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

def show_theta_trajectory():
  plt.figure('show_theta_trajectory')
  sampledata = make_sample_data()
  coords = sampledata['coords']
  transdyn = sampledata['trans_dyn']
  traj = sampledata['traj']
  transpar = sampledata['trans_par']
  theta = compute_theta(traj.phase, transpar)
  plt.plot(traj.time, theta)
  plt.xlabel(R't', fontsize=16)
  plt.ylabel(R'$\theta$', fontsize=16)
  plt.grid(True)
  set_pi_yticks('1/4')

  plt.figure('projection onto pi plane:')
  traj = sampledata['traj']
  trans_par = sampledata['trans_par']
  x0 = trans_par.proj_plane_origin
  pi_x = trans_par.proj_plane_x
  pi_y = trans_par.proj_plane_y

  x = (traj.phase - x0) @ pi_x
  y = (traj.phase - x0) @ pi_y
  singpt = x0[0:2]
  plt.plot(x, y)
  plt.grid(True)

  fig, axes = plt.subplots(2, 1, sharex=True, num='trajectory wrt theta')
  plt.sca(axes[0])
  plt.plot(theta, traj.coords)
  plt.axhline(singpt[0], ls='--', color='grey')
  plt.axhline(singpt[1], ls='--', color='grey')
  plt.grid(True)
  plt.sca(axes[1])
  plt.plot(theta, traj.vels)
  set_pi_xticks('1/2')
  plt.grid(True)

  plt.tight_layout()

def show_trasnverse_linearization_coefs():
  sampledata = make_sample_data()
  transdyn = sampledata['trans_dyn']
  theta = np.linspace(transdyn.transverse_coords.theta_min, transdyn.transverse_coords.theta_max, 400)
  A = np.array([transdyn.A_fun(e) for e in theta])
  B = np.array([transdyn.B_fun(e) for e in theta])

  fig, axes = plt.subplots(2, 1, sharex=True)
  ax = axes[0]
  plt.sca(ax)
  plt.title('matrix A')
  plt.plot(theta, A[:,:,0])
  plt.plot(theta, A[:,:,1])
  plt.plot(theta, A[:,:,2])
  plt.legend([R'$A_{11}$', R'$A_{21}$', R'$B_{31}$',
              R'$A_{12}$', R'$A_{22}$', R'$B_{32}$',
              R'$A_{13}$', R'$A_{23}$', R'$B_{33}$'])
  plt.grid(True)

  plt.sca(ax)
  ax = axes[1]
  plt.sca(ax)
  plt.plot(theta, B[:,:,0])
  plt.legend([R'$B_1$', R'$B_2$', R'$B_3$'])
  set_pi_xticks('1/2')
  plt.grid(True)

  plt.tight_layout()

def main():
  verify_free_motion_transverse_dynamics()
  plt.pause(0.001)
  verify_transverse_dynamics()
  plt.pause(0.001)
  test_transverse_linearization()
  plt.pause(0.001)
  show_theta_trajectory()
  plt.pause(0.001)
  show_trasnverse_linearization_coefs()
  plt.show()

if __name__ == '__main__':
  main()
