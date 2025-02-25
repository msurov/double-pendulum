import casadi as ca


class MechanicalSystem:
  def __init__(self):
    self.q = ca.MX.zeros(2)
    self.dq = ca.MX.zeros(2)
    self.u = ca.MX.zeros(1)

    self.M_expr = ca.MX.zeros(2, 2)
    self.C_expr = ca.MX.zeros(2, 2)
    self.G_expr = ca.MX.zeros(2, 1)
    self.B_expr = ca.MX.zeros(2, 1)
    self.U_expr = ca.MX.zeros(1)
    self.K_expr = ca.MX.zeros(1)

    self.ddq_expr = ca.MX.zeros(2)
    self.rhs_expr = ca.MX.zeros(4)

    self.M = lambda q: None
    self.C = lambda q, dq: None
    self.G = lambda q: None
    self.B = lambda q: None
    self.U = lambda q: None
    self.K = lambda q, dq: None
    self.rhs = lambda state, u: None
