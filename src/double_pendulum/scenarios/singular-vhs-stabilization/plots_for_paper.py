import matplotlib.pyplot as plt
from common.mechsys import MechanicalSystem
from common.plots import set_pi_xticks
from common.numpy_utils import integrate_array, cont_angle
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from typing import Tuple
from double_pendulum.scenarios.plot_phase import plot_singular_phase_portrait

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
from double_pendulum.scenarios.transverse_feedback_closed_loop_sim import (
  DoublePendulumTransverseFeedback,
  TranverseFeedbackController,
  TransverseDynamics,
  TranverseFeedbackControllerPar,
  TransverseDynamics,
)
from common.linsys import solve_gramian_mat
from common.numpy_utils import (
  min_eigval,
  map_array
)
from double_pendulum.anim import (
  motion_schematic,
  motion_schematic_v2,
  DoublePendulumViewParam,
  get_view_parameters
)
from singular_motion_planner.singular_constrs import get_sing_constr_at
from scipy.integrate import solve_ivp

from transverse_dynamics.cylindrical_transverse_coordinates import (
  CylindricalTransverseCoordinates,
  CylindricalTransverseCoordinatesPar,
)


def make_constr(par : DoublePendulumParam, q_sing : np.ndarray, scale=1):
  B = ca.DM([[1], [0]])
  B_perp = ca.DM([[0, -1]])

  par2 = convert_parameters(par)
  p2 = par2.p2
  p3 = par2.p3
  c1 = ca.DM([[p3], [-p2 * ca.cos(q_sing[1]) - p3]])
  k = ca.sqrt(2) / 2 * p2 * p3 - 1.05 * p2**2 / 3
  c2 = B_perp.T

  print('VHC k: ', k)
  print('VHC c1: ', c1)
  print('VHC c2: ', c2)

  phi = ca.SX.sym('phi')
  constr_expr = q_sing + c1 * phi + c2 * k * phi**2 / 2
  constr_fun = ca.Function('constr', [phi], [constr_expr])
  return constr_fun

def compute_autonomous_fund_mat(trans_dyn, interval, **integ_args):
  A0 = trans_dyn.A_fun(0)
  n,_ = A0.shape

  def rhs(t, st):
    X = np.reshape(st, (n, n))
    dX = trans_dyn.A_fun(t) @ X
    dx = np.reshape(dX, (n**2,))
    return dx
  
  X0 = np.eye(n)
  x0 = np.reshape(X0, (n**2,))
  sol = solve_ivp(rhs, interval, x0, **integ_args)
  npts, = sol.t.shape
  F = sol.y.T.reshape((npts, n, n))
  return sol.t, F

def compute_traj_data(par):
  dynamics = DoublePendulumDynamics(par)
  singpt = [-np.pi/4, 3*np.pi/4]
  constr = make_constr(par, singpt)
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-2, -0.5e-3], 0.0, max_step=1e-3)
  tr_right = solve_reduced(reduced, [2., 0.5e-3], 0.0, max_step=1e-3)
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
  trans_dyn = TransverseDynamics(dynamics, coords)

  return {
    'reduced_traj': tr_closed,
    'traj': tr_orig,
    'dynamics': dynamics,
    'reduced_dynamics': reduced,
    'transverse_coords': coords,
    'transverse_dynamics': trans_dyn,
  }

def plot_trajectories():
  par = double_pendulum_param_default
  data = compute_traj_data(par)
  traj = data['traj']
  reduced = data['reduced_dynamics']
  traj_reduced = data['reduced_traj']

  fig = show_trajectory_projections(traj)
  plt.savefig('fig/pendubot-reference-trajectory-projections.pdf')
  fig = show_reduced_dynamics_phase_prortrait(reduced, traj_reduced)
  plt.savefig('fig/pendubot-phase-portrait.pdf')
  view_par = get_view_parameters(par)
  fig = motion_schematic(traj, view_par)
  fig.gca().axes.xaxis.set_ticklabels([])
  fig.gca().axes.yaxis.set_ticklabels([])
  fig.tight_layout()
  fig.savefig('fig/pendubot-oscillations-schematically.pdf')
  plt.show()

def compute_ltv_data(data):
  trans_dyn = data['transverse_dynamics']
  coords = data['transverse_coords']
  theta, W, F = solve_gramian_mat(trans_dyn.A_fun, trans_dyn.B_fun, [0, 2*np.pi], max_step=1e-3)
  W_min = map_array(min_eigval, W, 1)
  A = map_array(trans_dyn.A_fun, theta)
  B = map_array(trans_dyn.B_fun, theta)  
  return {
    'gramian': W,
    'gramian_min_eigval': W_min,
    'theta': theta,
    'A': A,
    'B': B
  }

def plot_linear_system_components():
  par = double_pendulum_param_default
  data = compute_traj_data(par)
  trans_dyn = data['transverse_dynamics']

  ltv = compute_ltv_data(data)
  A = ltv['A']
  B = ltv['B']
  theta = ltv['theta']
  Gmin = ltv['gramian_min_eigval']

  legend_par = {
    'loc': 'lower right',
    'fontsize': 12
  }

  fig, axes = plt.subplots(5, 1, sharex=True, figsize=(6, 6), num='linear system components')
  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(theta, A[:,:,0] @ np.diag([10, 1, 1]))
  plt.legend([R'$10 \cdot A_{11}$', R'$A_{21}$', R'$A_{31}$'], **legend_par)
  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(theta, A[:,:,1])
  plt.legend([R'$A_{12}$', R'$A_{22}$', R'$A_{32}$'], **legend_par)
  plt.sca(axes[2])
  plt.grid(True)
  plt.plot(theta, A[:,:,2] @ np.diag([10, 1, 1]))
  plt.legend([R'$10 \cdot A_{13}$', R'$A_{23}$', R'$A_{33}$'], **legend_par)
  plt.sca(axes[3])
  plt.grid(True)
  plt.plot(theta, B[:,:,0] @ np.diag([10, 10, 1]))
  plt.legend([R'$10 \cdot B_1$', R'$10 \cdot B_2$', R'$B_3$'], **legend_par)

  plt.sca(axes[4])
  plt.grid(True)
  plt.plot(theta, Gmin)
  plt.legend([R'$\min \mathrm{eigval} (G)$'], **legend_par)
  # plt.yticks([0, 5e-5, 10e-5], [R'$0$', R'$5 \cdot 10^{-5}$', R'$10^{-4}$'])

  set_pi_xticks('1/4', fontsize=16)
  plt.tight_layout(h_pad=-0.5)
  plt.savefig('fig/linsys-components.pdf')

  plt.show()

def compute_closed_loop_fund_mat(Afun, Bfun, Kfun, interval, **integ_args):
  n,_ = Afun(0).shape

  def rhs(t, x):
    X = np.reshape(x, (n, n))
    dX = (Afun(t) + Bfun(t) @ Kfun(t)) @ X
    return np.reshape(dX, (n**2))
  
  x0 = np.eye(n).reshape(n**2)
  sol = solve_ivp(rhs, interval, x0, **integ_args)
  npts, = sol.t.shape
  F = sol.y.T.reshape((npts, n, n))
  return sol.t, F

def compute_feedback(data):
  trans_par = TranverseFeedbackControllerPar(
    Q = np.diag([5., 2., 2.]),
    R = 1e-2 * np.eye(1),
    nsteps = 200
  )
  trans_dyn = data['transverse_dynamics']
  fb = DoublePendulumTransverseFeedback(trans_dyn, trans_par)

  theta = np.linspace(0, 2*np.pi, 400)
  K = fb.trans_feedback.Ksp(theta)
  P = np.array([fb.trans_feedback.Psp(w) for w in theta], float)
  A = np.array([trans_dyn.A_fun(w) for w in theta], float)
  B = np.array([trans_dyn.B_fun(w) for w in theta], float)

  _, F = compute_closed_loop_fund_mat(trans_dyn.A_fun, trans_dyn.B_fun, fb.trans_feedback.Ksp, [theta[0], theta[-1]], t_eval=theta)

  return {
    'trans_par': trans_par,
    'theta': theta,
    'K': K,
    'P': P,
    'A': A,
    'B': B,
    'F': F,
  }

def plot_feedback_coefs():
  par = double_pendulum_param_default
  data = compute_traj_data(par)
  fb_data = compute_feedback(data)

  theta = fb_data['theta']
  F = fb_data['F']
  F_eigvals = np.linalg.eigvals(F[-1,:,:])
  print('singular values of the monodromy matrix:', F_eigvals)

  trans_par = fb_data['trans_par']
  print('matrix Q:', trans_par.Q)
  print('matrix R:', trans_par.R)

  legend_par = {
    'loc': 'lower right',
    'fontsize': 12,
    'ncol': 3
  }

  fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6, 2), num='feedback coefficients')
  plt.grid(True)
  K = fb_data['K']
  plt.plot(theta, K[:,0,:])
  plt.legend([R'$K_1$', R'$K_2$', R'$K_3$'], **legend_par)
  set_pi_xticks('1/4', fontsize=16)
  plt.tight_layout()
  plt.savefig('fig/feedback-coefficients.pdf')
  plt.show()

def add_annotation(text : str, textpos : Tuple[int, int]):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.8
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': 18
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def plot_nonlin_simulation_results():
  par = double_pendulum_param_default
  data = compute_traj_data(par)
  sim_data = np.load('data/closed-loop-simulation-result.npy', allow_pickle=True).item()

  orig_traj = data['traj']
  sim_traj = sim_data.traj

  ref_curve_par = {
    'lw': 2,
    'color': 'red', 
    'ls': '--'
  }
  real_curve_par = {
    'alpha': 0.6,
    'color': 'darkblue',
    'lw': 0.8
  }

  fig, axes = plt.subplots(2, 2, sharex=True, num='closed loop sim', figsize=(6, 4))

  plt.sca(axes[0,0])
  plt.grid(True)
  plt.plot(sim_traj.coords[:,0], sim_traj.vels[:,0], **real_curve_par)
  plt.plot(orig_traj.coords[:,0], orig_traj.vels[:,0], **ref_curve_par)
  add_annotation(R'$\dot q_1$', [-0.25, 0.57])

  plt.sca(axes[0,1])
  plt.grid(True)
  plt.plot(sim_traj.coords[:,0], sim_traj.vels[:,1], **real_curve_par)
  plt.plot(orig_traj.coords[:,0], orig_traj.vels[:,1], **ref_curve_par)
  add_annotation(R'$\dot q_2$', [-0.18, 0.50])

  plt.sca(axes[1,0])
  plt.grid(True)
  n = orig_traj.time.shape[0] // 2
  plt.plot(sim_traj.coords[:,0], sim_traj.coords[:,1], **real_curve_par)
  plt.plot(orig_traj.coords[:n,0], orig_traj.coords[:n,1], **ref_curve_par)
  add_annotation(R'$q_2$', [-0.25, 0.57])
  add_annotation(R'$q_1$', [0.70, -0.18])

  plt.sca(axes[1,1])
  plt.grid(True)
  n = orig_traj.time.shape[0] // 2
  plt.plot(sim_traj.coords[:,0], sim_traj.control, **real_curve_par)
  plt.plot(orig_traj.coords[:n,0], orig_traj.control[:n], **ref_curve_par)
  add_annotation(R'$u$', [-0.18, 0.50])
  add_annotation(R'$q_1$', [0.70, -0.18])

  plt.tight_layout(h_pad=-0.1)
  plt.savefig('fig/nonlin-sim-results.pdf')

  n, = sim_traj.time.shape
  n = n // 2  
  w = sim_data.ctrl_state[:n,0]
  theta = sim_data.ctrl_state[:n,1]
  cont_angle(theta)
  xi = sim_data.ctrl_state[:n,2:5]
  V = sim_data.ctrl_state[:n,5]
  time = sim_traj.time[:n]
  u = sim_data.traj.control[:n]

  fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 4), num='transverse coordinates')
  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(time, xi)
  plt.legend([R'$\xi_1$', R'$\xi_2$', R'$\xi_3$'], loc='lower right', fontsize=14)

  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(time, V)
  plt.legend([R'$V$'], loc='upper right', fontsize=14)

  plt.sca(axes[2])
  plt.grid(True)
  plt.plot(time, w)
  plt.legend([R'$w$'], loc='upper right', fontsize=14)
  plt.xlabel('time')
  plt.tight_layout(h_pad=-0.2)
  plt.savefig('fig/transverse-coords.pdf')

  plt.show()

def plot_phase_portrait():
  par = double_pendulum_param_default
  data = compute_traj_data(par)
  reduced = data['reduced_dynamics']
  traj = data['reduced_traj']
  fig, ax = plt.subplots(1, 1, figsize=(6, 4), num='reduced phase portrait')
  plot_singular_phase_portrait(reduced, traj, 0., npts_x=30, npts_y=12, wider=0.1, higher=0.3)
  plt.savefig('fig/pendubot-phase-portrait.pdf')

if __name__ == '__main__':
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })

  plot_trajectories()
  plot_linear_system_components()
  plot_feedback_coefs()
  plot_nonlin_simulation_results()
  plot_phase_portrait()
