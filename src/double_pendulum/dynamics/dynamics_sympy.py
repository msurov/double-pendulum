import numpy as np
import sympy as sy
from double_pendulum.dynamics.parameters import DoublePendulumParam
from typing import List, Tuple

sin = sy.sin
cos = sy.cos
Symbol = sy.Symbol
Matrix = sy.Matrix
pprint = sy.pprint

def get_links_positions(par : DoublePendulumParam, thetas : Tuple[Symbol]) -> List[Symbol]:
  R"""
    compute (x,y) positions of links mass centers
  """
  l1, _ = par.lengths
  c1, c2 = par.mass_centers
  theta1,theta2 = thetas
  x1 = c1 * sin(theta1)
  y1 = c1 * cos(theta1)
  x2 = l1 * sin(theta1) + c2 * sin(theta1 + theta2)
  y2 = l1 * cos(theta1) + c2 * cos(theta1 + theta2)
  p1 = Matrix([[x1], [y1]])
  p2 = Matrix([[x2], [y2]])
  return [p1, p2]

def get_links_orientations(par : DoublePendulumParam, thetas : Tuple[Symbol]) -> List[Symbol]:
  R"""
    compute angular positions of links mass centers
  """
  return Matrix([thetas[0]]), Matrix([thetas[0] + thetas[1]])

def get_kinetic_energy_mat(par : DoublePendulumParam, thetas : Tuple[Symbol]) -> Symbol:
  R"""
    compute kinetic energy matrix
  """
  m1, m2 = par.masses
  I1, I2 = par.inertia

  p1, p2 = get_links_positions(par, thetas)
  Jac1 = p1.jacobian(thetas)
  Jac2 = p2.jacobian(thetas)
  M1 = Jac1.T @ Jac1 * m1
  M2 = Jac2.T @ Jac2 * m2

  a1, a2 = get_links_orientations(par, thetas)
  Jac3 = a1.jacobian(thetas)
  Jac4 = a2.jacobian(thetas)
  M3 = Jac3.T @ Jac3 * I1
  M4 = Jac4.T @ Jac4 * I2
  M = M1 + M2 + M3 + M4
  M.simplify()
  return M

def get_potential_energy(par : DoublePendulumParam, thetas : Tuple[Symbol]) -> Symbol:
  p1, p2 = get_links_positions(par, thetas)
  m1, m2 = par.masses
  g = par.gravity_accel
  U1 = p1[1,0] * m1 * g
  U2 = p2[1,0] * m2 * g
  U = U1 + U2
  U.simplify()
  return U

def get_gravity_mat(U : Symbol, thetas : Tuple[Symbol]) -> Matrix:
  U = Matrix([U])
  G = U.jacobian(thetas).T
  G.simplify()
  return G

def get_control_mar(par : DoublePendulumParam) -> Matrix:
  B = sy.zeros(2,1)
  B[par.actuated_joint,0] = 1
  return B

def get_coriolis_mat(M : Matrix, thetas : Tuple[Symbol], dq : Tuple[Symbol]) -> Matrix:
  dq = Matrix([dq]).T
  Mdq = M @ dq
  J = Mdq.jacobian(thetas)
  C = J - J.T / 2
  C.simplify()
  return C

class DoublePendulumDynamics:
  def __init__(self, par : DoublePendulumParam):
    q = sy.symbols(R'q_(1:3)', real=True)
    dq = sy.symbols(R'\dot{q}_(1:3)', real=True)
    u = sy.symbols('u', real=True)
    M = get_kinetic_energy_mat(par, q)
    U = get_potential_energy(par, q)
    G = get_gravity_mat(U, q)
    B = get_control_mar(par)
    C = get_coriolis_mat(M, q, dq)
    dq_ = sy.Matrix(dq)
    K = (dq_.T @ M @ dq_)[0,0] / 2
    K = K.simplify()

    self.u = u
    self.q = q
    self.dq = dq
    self.M_expr = M
    self.C_expr = C
    self.G_expr = G
    self.B_expr = B
    self.U_expr = U
    self.K_expr = K

    self.M = sy.lambdify([q], M)
    self.C = sy.lambdify([q, dq], C)
    self.G = sy.lambdify([q], G)
    self.B = sy.lambdify([q], B)
    self.U = sy.lambdify([q], U)
    self.K = sy.lambdify([q, dq], K)

    ddq = M.inv() @ (-C @ sy.Matrix(dq) - G + B * u)
    ddq.simplify()
    rhs = sy.zeros(4, 1)
    rhs[0:2,0] = dq
    rhs[2:4,0] = ddq
    self.rhs_expr = rhs
    state = (*q, *dq)
    self.rhs = sy.lambdify([state], rhs)

def get_sym_par() -> DoublePendulumParam:
  m1, m2 = sy.symbols('m_(1:3)', real=True, positive=True)
  c1, c2 = sy.symbols('c_(1:3)', real=True, positive=True)
  l1, l2 = sy.symbols('l_(1:3)', real=True, positive=True)
  I1, I2 = sy.symbols('I_(1:3)', real=True, positive=True)
  g = sy.symbols('g', real=True, positive=True)

  return DoublePendulumParam(
    [l1, l2],
    [c1, c2],
    [m1, m2],
    [I1, I2],
    0,
    g
  )

def get_symbolic_dynamics():
  par = get_sym_par()
  d = DoublePendulumDynamics(par)
  return {
    'M': d.M_expr,
    'C': d.C_expr,
    'G': d.G_expr,
    'B': d.B_expr,
    'U': d.U_expr,
    'K': d.K_expr,
    'q': d.q,
    'dq': d.dq,
    **par.todict()
  }

def show_equations():
  d = get_symbolic_dynamics()
  print('Kinetic energy matrix:')
  print('  M =', sy.latex(d['M']))
  print('Coriolis and centrifugal forces matrix:')
  print('  C =', sy.latex(d['C']))
  print('Gravity matrix:')
  print('  G =', sy.latex(d['G']))
  print('Control matrix:')
  print('  B =', sy.latex(d['B']))
  print('Potential energy:')
  print('  U =', sy.latex(d['U']))
  print('Kinetic energy:')
  print('  K =', sy.latex(d['K']))
  print('parameters:')
  print('  first link length:', *d['lengths'])
  print('  links mass center displacements:', *d['mass_centers'])
  print('  links masses:', *d['masses'])
  print('  links inertia momentum:', *d['inertia'])
  print('  gravity acceleration:', d['gravity_accel'])
