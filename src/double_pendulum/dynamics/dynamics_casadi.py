import casadi as ca
from double_pendulum.dynamics.parameters import DoublePendulumParam
from typing import List
from common.mechsys import MechanicalSystem

sin = ca.sin
cos = ca.cos
SX = ca.SX
jacobian = ca.jacobian

def get_links_positions(par : DoublePendulumParam, thetas : SX) -> List[SX]:
  R"""
    compute (x,y) positions of links mass centers
  """
  l1, _ = par.lengths
  c1, c2 = par.mass_centers
  theta1 = thetas[0]
  theta2 = thetas[1]
  x1 = c1 * sin(theta1)
  y1 = c1 * cos(theta1)
  x2 = l1 * sin(theta1) + c2 * sin(theta1 + theta2)
  y2 = l1 * cos(theta1) + c2 * cos(theta1 + theta2)
  p1 = ca.vertcat(x1, y1)
  p2 = ca.vertcat(x2, y2)
  return [p1, p2]

def get_links_orientations(par : DoublePendulumParam, thetas : SX) -> List[SX]:
  R"""
    compute angular positions of links mass centers
  """
  return [thetas[0], thetas[0] + thetas[1]]

def get_kinetic_energy_mat(par : DoublePendulumParam, thetas : SX) -> SX:
  R"""
    compute kinetic energy matrix
  """
  m1, m2 = par.masses
  I1, I2 = par.inertia

  p1, p2 = get_links_positions(par, thetas)
  Jac1 = jacobian(p1, thetas)
  Jac2 = jacobian(p2, thetas)
  M1 = Jac1.T @ Jac1 * m1
  M2 = Jac2.T @ Jac2 * m2

  a1, a2 = get_links_orientations(par, thetas)
  Jac3 = jacobian(a1, thetas)
  Jac4 = jacobian(a2, thetas)
  M3 = Jac3.T @ Jac3 * I1
  M4 = Jac4.T @ Jac4 * I2

  return M1 + M2 + M3 + M4

def get_potential_energy(par : DoublePendulumParam, thetas : SX) -> SX:
  p1, p2 = get_links_positions(par, thetas)
  m1, m2 = par.masses
  g = par.gravity_accel
  U1 = p1[1] * m1 * g
  U2 = p2[1] * m2 * g
  return U1 + U2

def get_gravity_mat(U : SX, thetas : SX) -> SX:
  G = jacobian(U, thetas).T
  return G

def get_control_mar(par : DoublePendulumParam) -> SX:
  B = SX.zeros(2,1)
  B[par.actiated_joint, 0] = 1
  return B

def get_coriolis_mat(M : SX, q : SX, dq : SX) -> SX:
  # Mdq = M @ dthetas
  # J = jacobian(Mdq, thetas)
  # C = J - J.T / 2
  C1 = ca.jtimes(M, q, dq)
  C2 = 0.5 * ca.jacobian(M @ dq, q).T
  C = C1 - C2
  C = ca.simplify(C)
  return C

class DoublePendulumDynamics(MechanicalSystem):
  R"""
    Symbolic dynamics of double pendulum system
    based on CasADi symbolic package
  """
  def __init__(self, par : DoublePendulumParam):
    q = ca.SX.sym('q', 2)
    dq = ca.SX.sym('dq', 2)
    u = ca.SX.sym('u', 1)
    M = get_kinetic_energy_mat(par, q)
    U = get_potential_energy(par, q)
    G = get_gravity_mat(U, q)
    B = get_control_mar(par)
    C = get_coriolis_mat(M, q, dq)
    K = dq.T @ M @ dq / 2
    E = U + K

    self.q = q
    self.dq = dq
    self.u = u
    self.M_expr = M
    self.C_expr = C
    self.G_expr = G
    self.B_expr = B
    self.U_expr = U
    self.K_expr = K
    self.E_expr = E

    state = ca.vertcat(q, dq)
    self.ddq_expr = ca.pinv(M) @ (-C @ dq - G + B @ u)
    self.rhs_expr = ca.vertcat(dq, self.ddq_expr)

    self.M = ca.Function('M', [q], [self.M_expr])
    self.C = ca.Function('C', [q, dq], [self.C_expr])
    self.G = ca.Function('G', [q], [self.G_expr])
    self.B = ca.Function('B', [q], [self.B_expr])
    self.U = ca.Function('U', [q], [self.U_expr])
    self.K = ca.Function('K', [q, dq], [self.K_expr])
    self.E = ca.Function('E', [q, dq], [self.E_expr])
    self.rhs = ca.Function('RHS', [state, u], [self.rhs_expr])
