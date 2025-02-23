from double_pendulum.dynamics import (
  DoublePendulumDynamics,
)
import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from common.trajectory import Trajectory


class ReducedDynamics:
  def __init__(self, dynamics : DoublePendulumDynamics, constr : ca.Function):
    s = ca.SX.sym('s')

    Q = constr(s)
    dQ = ca.jacobian(Q, s)
    ddQ = ca.jacobian(dQ, s)

    M = dynamics.M(Q)
    C = dynamics.C(Q, dQ)
    G = dynamics.G(Q)
    B = dynamics.B(Q)
    B_perp = ca.DM([[0, 1]])

    self.alpha_expr = B_perp @ M @ dQ
    self.dalpha_expr = ca.jacobian(self.alpha_expr, s)
    self.beta_expr = B_perp @ (M @ ddQ + C @ dQ)
    self.gamma_expr = B_perp @ G
    self.dgamma_expr = ca.jacobian(self.gamma_expr, s)

    self.s = s
    self.alpha = ca.Function('alpha', [self.s], [self.alpha_expr])
    self.dalpha = ca.Function('dalpha', [self.s], [self.dalpha_expr])
    self.beta = ca.Function('beta', [self.s], [self.beta_expr])
    self.gamma = ca.Function('gamma', [self.s], [self.gamma_expr])
    self.dgamma = ca.Function('dgamma', [self.s], [self.dgamma_expr])

def compute_time(s, ds):
  dt = 2 * np.diff(s) / (ds[1:] + ds[:-1])
  t = np.zeros(len(s))
  t[1:] = np.cumsum(dt)
  return t

def solve_periodic(rd : ReducedDynamics, s0, smax, eps=1e-6, **solver_args) -> Trajectory:
  def rhs(s, y):
    dy = (-2 * rd.beta(s) * y - rd.gamma(s)) / rd.alpha(s)
    return float(dy)
  
  def stop_condition(s, y):
    if y[0] <= 0 and abs(s - s0) > eps:
      return -1
    return 1

  stop_condition.terminal = True
  sol = solve_ivp(rhs, [s0, smax], [0.], **solver_args, events=stop_condition)

  if len(sol.t) <= 2:
    return None

  y = np.clip(sol.y[0], 0., np.inf)
  s = sol.t

  ds = np.sqrt(2 * y)
  ds_full = np.concatenate((ds, -ds[-2::-1]))
  s_full = np.concatenate((s, s[-2::-1]))

  t = compute_time(s_full, ds_full)
  return Trajectory(
    time = t,
    phase = np.array([s_full, ds_full]).T,
    control = None
  )

def filter_out_duplicates(x, y, eps=1e-9):
  mask = np.ones(x.shape, bool)
  mask[:-1] = np.abs(np.diff(x)) > eps
  x = x[mask]
  y = y[mask,...]
  return x, y

def solve_reduced(rd : ReducedDynamics, sdiap, ds0, **solver_args) -> Trajectory:
  def rhs(s, y):
    dy = (-2 * rd.beta(s) * y - rd.gamma(s)) / rd.alpha(s)
    return float(dy)

  def stop(s, y):
    if y[0] <= 0 and abs(s - sdiap[0]) > eps:
      print('ZZZZZZZZZZZ')
      return 0.
    return 1.

  eps = 1e-6
  stop.terminal = True
  y0 = ds0**2/2
  # sol = solve_ivp(rhs, sdiap, [y0], **solver_args, events=stop)
  sol = solve_ivp(rhs, sdiap, [y0], **solver_args)
  s, y = filter_out_duplicates(sol.t, sol.y[0])

  if len(s) <= 3:
    return None

  if y[-1] < 0:
    y1,y2 = y[-2:]
    assert y1 >= 0
    s1,s2 = s[-2:]
    s_final = (y2 * s1 - y1 * s2) / (y2 - y1)
    s[-1] = s_final
    y[-1] = 0.

  ds = np.sqrt(2 * y)
  t = compute_time(s, ds)

  return Trajectory(
    time = t,
    phase = np.array([s, ds]).T,
    control = None
  )

def reconstruct_trajectory(constr : ca.Function, reduced : ReducedDynamics, 
                           dynamics : DoublePendulumDynamics,
                           reduced_traj : Trajectory) -> Trajectory:

  s_expr = reduced.s
  constr_expr = constr(s_expr)
  Dconstr_expr = ca.jacobian(constr_expr, s_expr)
  DDconstr_expr = ca.jacobian(Dconstr_expr, s_expr)
  ds_expr = ca.SX.sym('ds')
  dq_expr = Dconstr_expr * ds_expr
  dds_expr = (-reduced.beta_expr * ds_expr**2 - reduced.gamma_expr) / reduced.alpha_expr
  ddq_expr = Dconstr_expr * dds_expr + DDconstr_expr * ds_expr**2
  Bt = dynamics.B(constr_expr).T
  u_expr = Bt @ (dynamics.M(constr_expr) @ ddq_expr + dynamics.C(constr_expr, dq_expr) @ dq_expr + dynamics.G(constr_expr)) / (Bt @ Bt.T)
  u_fun = ca.Function('u', [s_expr, ds_expr], [u_expr])
  dq_fun = ca.Function('dq', [s_expr, ds_expr], [dq_expr])

  nq,_ = constr(0.).shape
  nt, = reduced_traj.time.shape
  state = np.zeros((nt, 2 * nq))
  u = np.zeros(nt)

  for i in range(nt):
    s = reduced_traj.coords[i]
    ds = reduced_traj.vels[i]
    q = constr(s)
    dq = dq_fun(s, ds)
    u[i] = float(u_fun(s, ds))
    state[i,0:nq] = np.reshape(q, (-1,))
    state[i,nq:] = np.reshape(dq, (-1,))

  return Trajectory(
    time = reduced_traj.time,
    phase = state,
    control = u
  )
