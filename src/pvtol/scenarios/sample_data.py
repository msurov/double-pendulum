import numpy as np
import casadi as ca
from scipy.interpolate import make_interp_spline
from pvtol.dynamics import PVTOLAircraftDynamics
from common.mechsys import MechanicalSystem
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
from common.numpy_utils import map_array, cont_angle
from transverse_dynamics.cylindrical_transverse_coordinates import (
  CylindricalTransverseCoordinates,
  CylindricalTransverseCoordinatesPar,
)
from transverse_dynamics.transverse_dynamics import TransverseDynamics


def get_sing_constr_at(dynamics : MechanicalSystem, q_sing : np.ndarray, D : np.ndarray, scale=1):
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
  constr_expr = q_sing + N_val * theta / scale + 0.5 * k * F_val * theta**2 / scale**2
  constr_fun = ca.Function('constr', [theta], [constr_expr])
  return constr_fun

def compute_45deg_oscillations(dynamics : PVTOLAircraftDynamics):
  singpt = np.array([0, 0, np.pi/4])
  constr = get_sing_constr_at(dynamics, singpt, [1., 2.5])

  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-0.15, -1e-3], 0.0, max_step=1e-3)
  tr_right = solve_reduced(reduced, [0.15, 1e-3], 0.0, max_step=1e-3)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)

  return {
    'reduced_dynamics': reduced,
    'reduced_traj': tr_closed,
    'constr': constr,
    'singpt': singpt
  }

def compute_circular_traj(dynamics : PVTOLAircraftDynamics):
  singpt = np.array([0, 0, np.pi/2])

  theta = ca.SX.sym('theta')
  constr_sym = ca.vertcat(
    theta,
    -0.5 * theta**2,
    ca.pi / 2 - ca.arctan(2 * theta)
  )
  constr = ca.Function('constr', [theta], [constr_sym])

  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-1., -1e-2], 0.0, max_step=1e-2)
  tr_right = solve_reduced(reduced, [1., 1e-2], 0.0, max_step=1e-2)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)

  return {
    'reduced_dynamics': reduced,
    'reduced_traj': tr_closed,
    'constr': constr,
    'singpt': singpt
  }

def compute_traj_data(traj_name : str, dynamics : PVTOLAircraftDynamics):
  match traj_name:
    case 'tictoc': return compute_circular_traj(dynamics)
    case '45deg': return compute_45deg_oscillations(dynamics)
    case _: assert False

def make_sample_data(traj_name : str):
  dynamics = PVTOLAircraftDynamics()
  result = compute_traj_data(traj_name, dynamics)
  reduced_traj = result['reduced_traj']
  reduced_dynamics = result['reduced_dynamics']
  constr = result['constr']
  singpt = result['singpt']
  tr_orig = reconstruct_trajectory(constr, reduced_dynamics, dynamics, reduced_traj)
  trans_par = CylindricalTransverseCoordinatesPar(
    transverse_projection_mat = np.array([
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1]
    ]),
    proj_plane_x = np.array([0, 0, 0, 1, 0, 0]),
    proj_plane_y = np.array([1, 0, 0, 0, 0, 0]),
    proj_plane_origin = np.concatenate((singpt, [0, 0, 0]))
  )
  trans_coords = CylindricalTransverseCoordinates(tr_orig, trans_par)
  traj_of_time = make_interp_spline(tr_orig.time, tr_orig.phase, k=5, bc_type='periodic')
  ctrl_of_time = make_interp_spline(tr_orig.time, tr_orig.control, k=3, bc_type='periodic')
  trans_dyn = TransverseDynamics(dynamics, trans_coords)
  
  tmp = map_array(trans_coords.forward_transform_fun, tr_orig.phase, (6,))
  theta = tmp[:,0]
  cont_angle(theta)

  traj_of_theta = make_interp_spline(theta, tr_orig.phase, k=5, bc_type='periodic')
  ctrl_of_theta = make_interp_spline(theta, tr_orig.control, k=3, bc_type='periodic')

  return {
    'dynamics': dynamics,
    'reduced': reduced_dynamics,
    'reduced_traj': reduced_traj,
    'constr': constr,
    'traj': tr_orig,
    'traj_of_theta': traj_of_theta,
    'ctrl_of_theta': ctrl_of_theta,
    'transverse_dynamics': trans_dyn,
    'transverse_coordinates': trans_coords
  }
