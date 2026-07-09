import casadi as ca
from cart_pendulum.dynamics.parameters import CartPendulumPar
from typing import List
from common.mechsys import MechanicalSystem


sin = ca.sin
cos = ca.cos
MX = ca.MX
jacobian = ca.jacobian

class CartPendulumDynamics(MechanicalSystem):
  R"""
    Symbolic dynamics of cart pendulum system
    based on CasADi symbolic package
  """
  def __init__(self, par : CartPendulumPar):
    q = ca.MX.sym('q', 2)
    dq = ca.MX.sym('dq', 2)
    u = ca.MX.sym('u', 1)

    mc = par.cart_mass
    mp = par.pendulum_mass
    l = par.pendulum_length
    g = par.gravity_accel

    M = MX.zeros((2,2))
    M[0,0] = mc + mp
    M[0,1] = \
    M[1,0] = mp * l * cos(q[1])
    M[1,1] = mp * l**2

    C = MX.zeros((2,2))
    C[0,1] = -mp * l * sin(q[1]) * dq[1]

    D = MX.zeros((2, 2))
    D[1,1] = 0 if par.joint_friction is None else par.joint_friction

    G = MX.zeros((2,1))
    G[1] = -mp * g * l * sin(q[1])

    U = mp * g * l * cos(q[1])

    B = MX.zeros((2,1))
    B[0] = 1

    B_perp = MX.zeros((1, 2))
    B_perp[0,1] = 1

    K = dq.T @ M @ dq / 2
    E = U + K

    self.q = q
    self.dq = dq
    self.u = u
    self.M_expr = M
    self.C_expr = C
    self.D_expr = D
    self.G_expr = G
    self.B_expr = B
    self.B_perp_expr = B_perp
    self.U_expr = U
    self.K_expr = K
    self.E_expr = E

    state = ca.vertcat(q, dq)
    self.ddq_expr = ca.pinv(M) @ (-C @ dq - D @ dq - G + B @ u)
    self.rhs_expr = ca.vertcat(dq, self.ddq_expr)

    self.M = ca.Function('M', [q], [self.M_expr])
    self.C = ca.Function('C', [q, dq], [self.C_expr])
    self.D = ca.Function('D', [q], [self.D_expr])
    self.G = ca.Function('G', [q], [self.G_expr])
    self.B = ca.Function('B', [q], [self.B_expr])
    self.B_perp = ca.Function('B_perp', [q], [self.B_perp_expr])
    self.U = ca.Function('U', [q], [self.U_expr])
    self.K = ca.Function('K', [q, dq], [self.K_expr])
    self.E = ca.Function('E', [q, dq], [self.E_expr])
    self.rhs = ca.Function('RHS', [state, u], [self.rhs_expr])
