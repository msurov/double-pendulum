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

def solve_reduced(rd : ReducedDynamics, sdiap, ds0, **solver_args) -> Trajectory:
  def rhs(s, y):
    dy = (-2 * rd.beta(s) * y - rd.gamma(s)) / rd.alpha(s)
    return float(dy)
  
  def event(s, y):
    if y[0] <= 0 and s != sdiap[0]:
      event.terminate = True
    return 0

  y0 = ds0**2/2
  sol = solve_ivp(rhs, sdiap, [y0], **solver_args, events=event)
  ds = np.sqrt(2 * sol.y[0])
  s = sol.t
  t = compute_time(s, ds)
  return Trajectory(
    time = t,
    phase = np.array([s, ds]).T,
    control = None
  )
