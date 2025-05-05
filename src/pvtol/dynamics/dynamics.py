import casadi as ca
from common.mechsys import MechanicalSystem
from common.casadi_utils import mat_sx


class PVTOLAircraftDynamics(MechanicalSystem):
  R"""
    PVTOL aircraft symbolic dynamics
  """
  def __init__(self):
    q = ca.SX.sym('q', 3)
    dq = ca.SX.sym('dq', 3)
    u = ca.SX.sym('u', 2)

    M = ca.DM_eye(3)
    C = ca.DM.zeros(3, 3)

    B = mat_sx([
      [-ca.sin(q[2]), 0],
      [ca.cos(q[2]), 0],
      [0, 1]
    ])
    B_perp = ca.horzcat(
      ca.cos(q[2]), ca.sin(q[2]), 0
    )
    G = ca.DM([
      0,
      1,
      0
    ])

    self.q = q
    self.dq = dq
    self.u = u
    self.M_expr = M
    self.C_expr = C
    self.G_expr = G
    self.B_expr = B
    self.B_perp_expr = B_perp

    self.phase = ca.vertcat(q, dq)
    self.ddq_expr = ca.solve(M, -C @ dq - G + B @ u)
    self.rhs_expr = ca.vertcat(dq, self.ddq_expr)

    self.M = ca.Function('M', [q], [self.M_expr])
    self.C = ca.Function('C', [q, dq], [self.C_expr])
    self.G = ca.Function('G', [q], [self.G_expr])
    self.B = ca.Function('B', [q], [self.B_expr])
    self.B_perp = ca.Function('B_perp', [q], [self.B_perp_expr])
    self.rhs = ca.Function('RHS', [self.phase, u], [self.rhs_expr])

    self.f_expr = ca.vertcat(
      dq,
      ca.solve(M, -C @ dq - G)
    )
    self.g_expr = ca.vertcat(
      ca.DM.zeros(3, 2),
      ca.solve(M, B)
    )

    self.f = ca.Function('f', [self.phase], [self.f_expr])
    self.g = ca.Function('g', [self.phase], [self.g_expr])
