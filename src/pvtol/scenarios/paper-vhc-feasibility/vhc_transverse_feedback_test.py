import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from common.numpy_utils import (
  cont_angle,
  map_array
)
from common.trajectory import Trajectory
from common.bspline_sym import (
  MXSpline,
  make_interp_spline
)
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
from common.linsys import (
  solve_gramian_mat,
  find_fund_mat
)
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from pvtol.dynamics import PVTOLAircraftDynamics
from transverse_dynamics.transverse_dynamics import (
  TransverseDynamics,
  verify_transversality
)
from transverse_dynamics.vhc_transverse_coordinates import VHCTransverseCoordinates
from transverse_dynamics.transverse_feedback import (
  TranverseFeedbackController,
  TranverseFeedbackControllerPar
)
from pvtol.scenarios.transient_process_plots import (
  show_transient,
  plot_transverse,
  plot_linsys
)
from pvtol.sim import (
  PVTOLAircraftSimulator,
  PVTOLAircraftSimulatorPar,
  PVTOLAircraftFeedback,
  SimulationResult
)


from pvtol.scenarios.sample_data import compute_circular_traj

def make_vhc1_parametric():
  singpt = np.array([0, 0, np.pi/2])
  theta = ca.SX.sym('theta')
  constr_sym = ca.vertcat(
    theta,
    -0.5 * theta**2,
    ca.pi / 2 - ca.arctan(2 * theta)
  )
  constr = ca.Function('constr', [theta], [constr_sym])
  return constr

def make_vhc1_implicit():
  q = ca.MX.sym('q', 3)
  x = q[0]
  z = q[1]
  psi = q[2]
  implicit_constr_expr = ca.vertcat(
    z + 0.5 * x**2,
    psi - ca.pi / 2 + ca.arctan(2 * x)
  )
  implicit_constr_fun = ca.Function('h', [q], [implicit_constr_expr])

  lam = ca.MX.sym('lam')
  y = ca.MX.sym('y', 2)
  inv_implicit_constr_expr = ca.vertcat(
    lam,
    y[0] - 0.5 * lam**2,
    y[1] + ca.pi / 2 - ca.arctan(2 * lam)
  )
  inv_implicit_constr_fun = ca.Function('h', [y, lam], [inv_implicit_constr_expr])

  return {
    'implicit': implicit_constr_fun,
    'implicit_inv': inv_implicit_constr_fun,
    'free_var_idx': 0,
    'free_var_origin': 0.
  }

def make_vhc1(dynamics : PVTOLAircraftDynamics):
  return {
    'parametric': make_vhc1_parametric(),
    **make_vhc1_implicit()
  }

def make_vhc2(dynamics : PVTOLAircraftDynamics):
  q_sing = ca.DM([0, 0, np.pi/4])
  D = ca.DM([1., 2.5])
  B = dynamics.B_expr
  B_perp = dynamics.B_perp_expr
  G = dynamics.G_expr

  gam = ca.evalf(ca.substitute(B_perp @ G, dynamics.q, q_sing))
  if abs(gam) < 1e-6:
    return None

  if gam < 0:
    B_perp = -B_perp

  q = dynamics.q
  dq = dynamics.dq
  M = dynamics.M_expr
  C = dynamics.C_expr

  N = ca.solve(M, B @ D)
  P = M @ B_perp.T
  F = B_perp.T / (B_perp @ M @ B_perp.T)

  left = B_perp @ C @ N
  left = ca.substitute(left, dq, N)
  left = ca.substitute(left, q, q_sing)
  left = ca.evalf(left)
  right = N.T @ ca.jtimes(P, dynamics.q, N)
  right = ca.substitute(right, q, q_sing)
  right = ca.evalf(right)

  if left >= right:
    return None

  k = -1/3 * left - 2/3 * right
  N_val = ca.evalf(ca.substitute(N, q, q_sing))
  F_val = ca.evalf(ca.substitute(F, q, q_sing))

  theta = ca.SX.sym('theta')
  constr_expr = q_sing + N_val * theta + 0.5 * k * F_val * theta**2
  constr_fun = ca.Function('constr', [theta], [constr_expr])

  q = ca.SX.sym('q', 3)
  free_var_idx = 2
  lam = q[free_var_idx]
  lam_sing = q_sing[free_var_idx]
  assert F_val[free_var_idx] == 0.
  theta = (lam - lam_sing) / N_val[free_var_idx]

  h_expr = q - (q_sing + N_val * theta + 0.5 * k * F_val * theta**2)
  h_expr = h_expr[0:2]
  h_fun = ca.Function('h', [q], [h_expr])

  y = ca.SX.sym('y', 2)
  h_inv_expr = ca.vertcat(y, 0) + q_sing + N_val * theta + 0.5 * k * F_val * theta**2
  h_inv_fun = ca.Function('h', [y, lam], [h_inv_expr])

  return {
    'parametric': constr_fun,
    'implicit': h_fun,
    'implicit_inv': h_inv_fun,
    'free_var_idx': free_var_idx,
    'free_var_origin': q_sing[free_var_idx]
  }

def compute_gramian(trans_dyn):
  t, W, F = solve_gramian_mat(trans_dyn.A_fun, trans_dyn.B_fun, [0, 2*np.pi], max_step=1e-2)
  evs = map_array(lambda w: np.sort(np.linalg.eigvals(w)), W)
  evals = np.linalg.eigvals(W[-1])
  evals.sort()
  evals = evals[::-1]
  print('gramian eigvals: ', np.array2string(evals, precision=8, separator=', '))

def compute_monodromy(trans_dyn, fb):
  A_closed_loop = lambda t: trans_dyn.A_fun(t) + trans_dyn.B_fun(t) @ fb.Ksp(t)
  t, F = find_fund_mat(A_closed_loop, [0, 2*np.pi], max_step=1e-2)
  FT = F[-1]
  evals = np.linalg.eigvals(FT)
  print('monodromy eigvals: ', np.array2string(evals, precision=8, separator=', '))

def trajectory1_scenario():
  dynamics = PVTOLAircraftDynamics()

  vhc = make_vhc1(dynamics)
  phi = vhc['parametric']

  reduced = ReducedDynamics(dynamics, phi)
  tr_left = solve_reduced(reduced, [-1., -1e-2], 0.0, max_step=1e-2)
  tr_right = solve_reduced(reduced, [1., 1e-2], 0.0, max_step=1e-2)
  tr_up = traj_join(tr_left, tr_right[::-1])
  reduced_traj = traj_forth_and_back(tr_up)
  ref_traj = reconstruct_trajectory(phi, reduced, dynamics, reduced_traj)

  h = vhc['implicit']
  hinv = vhc['implicit_inv']
  free_var_idx = vhc['free_var_idx']
  free_var_origin = vhc['free_var_origin']

  trans_coords = VHCTransverseCoordinates(ref_traj, h, hinv, free_var_idx, free_var_origin)
  verify_transversality(trans_coords)
  trans_dyn = TransverseDynamics(dynamics, trans_coords)
  compute_gramian(trans_dyn)

  tmp = trans_coords.forward_transform_fun(ref_traj.phase[12,:])
  theta = tmp[0]
  xi = tmp[1:]

  fb_par = TranverseFeedbackControllerPar(
    Q = np.eye(5),
    R = np.eye(2),
    nsteps = 1000,
    S = np.eye(5)
  )
  fb = TranverseFeedbackController(trans_dyn, fb_par)
  compute_monodromy(trans_dyn, fb)

  sim_par = PVTOLAircraftSimulatorPar(
    timestep = 1e-3,
    thrust_diap = [-20, 20],
    torque_diap = [-20, 20],
  )
  sim = PVTOLAircraftSimulator(sim_par, fb)

  simtime = 2. * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.array([0.1, -0.5, 0., 0., 0., 0.])
  simres = sim.run(x0, 0., simtime)

  fig = show_transient(ref_traj, simres.traj, simres.ctrl_state, trans_coords.usp)
  fig.savefig('fig/pvtol_phase_coords.pdf')
  fig = plot_transverse(simres, trans_coords.usp)
  fig.savefig('fig/pvtol_transverse.pdf')
  fig = plot_linsys(trans_dyn, fb)

  plt.show()

def trajectory2_scenario():
  dynamics = PVTOLAircraftDynamics()

  vhc = make_vhc2(dynamics)
  phi = vhc['parametric']

  reduced = ReducedDynamics(dynamics, phi)
  tr_left = solve_reduced(reduced, [-0.17, -1e-2], 0.0, max_step=1e-2)
  tr_right = solve_reduced(reduced, [0.17, 1e-2], 0.0, max_step=1e-2)
  tr_up = traj_join(tr_left, tr_right[::-1])
  reduced_traj = traj_forth_and_back(tr_up)
  ref_traj = reconstruct_trajectory(phi, reduced, dynamics, reduced_traj)

  h = vhc['implicit']
  hinv = vhc['implicit_inv']
  free_var_idx = vhc['free_var_idx']
  free_var_origin = vhc['free_var_origin']

  trans_coords = VHCTransverseCoordinates(ref_traj, h, hinv, free_var_idx, free_var_origin)
  verify_transversality(trans_coords)
  trans_dyn = TransverseDynamics(dynamics, trans_coords)

  tmp = trans_coords.forward_transform_fun(ref_traj.phase[12,:])
  theta = tmp[0]
  xi = tmp[1:]

  fb_par = TranverseFeedbackControllerPar(
    Q = 40 * np.eye(5),
    R = np.eye(2),
    nsteps = 1000,
    S = np.eye(5)
  )
  fb = TranverseFeedbackController(trans_dyn, fb_par)

  sim_par = PVTOLAircraftSimulatorPar(
    timestep = 1e-3,
    thrust_diap = [-20, 20],
    torque_diap = [-20, 20],
  )
  sim = PVTOLAircraftSimulator(sim_par, fb)

  simtime = 4. * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.array([0.05, -0.5, 0., 0., 0., 0.])
  simres = sim.run(x0, 0., simtime)

  fig = show_transient(ref_traj, simres.traj, simres.ctrl_state, trans_coords.usp)
  fig = plot_transverse(simres, trans_coords.usp)
  fig = plot_linsys(trans_dyn, fb)

  plt.show()

if __name__ == '__main__':
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  np.set_printoptions(suppress=True, precision=4)
  trajectory1_scenario()
  # trajectory2_scenario()