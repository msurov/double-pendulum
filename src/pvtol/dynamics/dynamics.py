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

    state = ca.vertcat(q, dq)
    self.ddq_expr = ca.pinv(M) @ (-C @ dq - G + B @ u)
    self.rhs_expr = ca.vertcat(dq, self.ddq_expr)

    self.M = ca.Function('M', [q], [self.M_expr])
    self.C = ca.Function('C', [q, dq], [self.C_expr])
    self.G = ca.Function('G', [q], [self.G_expr])
    self.B = ca.Function('B', [q], [self.B_expr])
    self.B_perp = ca.Function('B_perp', [q], [self.B_perp_expr])
    self.rhs = ca.Function('RHS', [state, u], [self.rhs_expr])
