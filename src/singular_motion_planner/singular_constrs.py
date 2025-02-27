from common.mechsys import MechanicalSystem
import casadi as ca
import numpy as np


def get_sing_constr_at(dynamics : MechanicalSystem, q_sing : np.ndarray, scale=1):
  J = ca.DM([
    [0, -1],
    [1, 0]
  ])
  B = dynamics.B_expr
  B_perp = (J @ B).T

  gam = ca.evalf(ca.substitute(B_perp @ dynamics.G_expr, dynamics.q, q_sing))
  if abs(gam) < 1e-6:
    return None

  if gam < 0:
    B_perp = -B_perp

  q = dynamics.q
  dq = dynamics.dq
  M = dynamics.M_expr
  C = dynamics.C_expr
  G = dynamics.G_expr

  N = ca.solve(M, B)
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
 