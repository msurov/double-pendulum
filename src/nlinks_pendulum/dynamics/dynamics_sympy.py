import numpy as np
import sympy as sy
from nlinks_pendulum.dynamics.parameters import NLinksPendulumParam
from typing import List, Tuple

sin = sy.sin
cos = sy.cos
Symbol = sy.Symbol
Matrix = sy.Matrix
pprint = sy.pprint


def get_links_positions(par : NLinksPendulumParam, q : Tuple[Symbol]) -> List[Symbol]:
  R"""
    compute (x,y) positions of links mass centers
  """
  nlinks = len(q)

  cur_joint = Matrix([0, 0])
  cur_angle = 0
  mass_centers = []
  joints = []

  for i in range(0, nlinks):
    cur_angle += q[i]
    direction = Matrix([
      sin(cur_angle),
      cos(cur_angle),
    ])
    mass_center = cur_joint + direction * par.mass_centers[i]
    next_joint = cur_joint + direction * par.lengths[i]
    joints.append(cur_joint)
    mass_centers.append(mass_center)
    cur_joint = next_joint
  
  return joints, mass_centers

def get_links_orientations(q : Tuple[Symbol]) -> List[Symbol]:
  R"""
    compute angular positions of links mass centers
  """
  nlinks = len(q)
  orientations = []
  angle = 0
  for i in range(nlinks):
    angle += q[i]
    orientations.append(angle)
  return orientations

def get_kinetic_energy_mat(par : NLinksPendulumParam, q : Tuple[Symbol]) -> Symbol:
  R"""
    compute kinetic energy matrix
  """
  _, mass_centers = get_links_positions(par, q)
  nlinks = len(q)
  M = sy.zeros(nlinks, nlinks)

  for c, m in zip(mass_centers, par.masses):
    J = c.jacobian(q)
    M += J.T @ J * m

  orients = get_links_orientations(q)

  for a, inertia in zip(orients, par.inertia):
    J = sy.Matrix([a]).jacobian(q)
    M += J.T @ J * inertia

  M.simplify()
  return M

def get_potential_energy(par : NLinksPendulumParam, q : Tuple[Symbol]) -> Symbol:
  _, mass_centers = get_links_positions(par, q)
  U = 0
  g = par.gravity_accel
  for c, m in zip(mass_centers, par.masses):
    U += m * g * c[1, 0]
  U.simplify()
  return U

def get_gravity_mat(U : Symbol, q : Tuple[Symbol]) -> Matrix:
  U = Matrix([U])
  G = U.jacobian(q).T
  G.simplify()
  return G

def get_control_mar(par : NLinksPendulumParam) -> Matrix:
  nlinks = len(par.lengths)
  B = sy.zeros(nlinks,1)
  B[par.actuated_joint,0] = 1
  return B

def get_coriolis_mat(M : Matrix, q : Tuple[Symbol], dq : Tuple[Symbol]) -> Matrix:
  dq = Matrix([dq]).T
  Mdq = M @ dq
  J = Mdq.jacobian(q)
  C = J - J.T / 2
  C.simplify()
  return C

class NLinksPendulumDynamics:
  def __init__(self, par : NLinksPendulumParam):
    nlinks = len(par.lengths)
    upper_bnound = str(nlinks + 1)
    q = sy.symbols(R'q_(1:' + upper_bnound + ')', real=True)
    dq = sy.symbols(R'\dot{q}_(1:' + upper_bnound + ')', real=True)
    u = sy.symbols('u', real=True)
    M = get_kinetic_energy_mat(par, q)
    U = get_potential_energy(par, q)
    G = get_gravity_mat(U, q)
    B = get_control_mar(par)
    C = get_coriolis_mat(M, q, dq)
    dq_ = sy.Matrix(dq)
    K = (dq_.T @ M @ dq_)[0,0] / 2
    # K = K.simplify()

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

    # ddq = M.solve(-C @ dq_ - G + B * u)

    # # ddq.simplify()
    # rhs = sy.zeros(2*nlinks, 1)
    # rhs[0:nlinks,0] = dq
    # rhs[nlinks:2*nlinks,0] = ddq
    # self.rhs_expr = rhs
    # state = (*q, *dq)
    # self.rhs = sy.lambdify([state], rhs)

def get_sym_par() -> NLinksPendulumParam:
  masses = sy.symbols('m_(1:4)', real=True, positive=True)
  mass_centers = sy.symbols('c_(1:4)', real=True, positive=True)
  lengths = sy.symbols('l_(1:4)', real=True, positive=True)
  inertia = sy.symbols('I_(1:4)', real=True, positive=True)
  g = sy.symbols('g', real=True, positive=True)

  return NLinksPendulumParam(
    lengths=lengths,
    mass_centers=mass_centers,
    masses=masses,
    inertia=inertia,
    actuated_joint=0,
    gravity_accel=g
  )

def get_symbolic_dynamics():
  par = get_sym_par()
  d = NLinksPendulumDynamics(par)
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

def test():
  par = NLinksPendulumParam(
    lengths=[2,2,2],
    mass_centers=[1,1,1],
    masses=[1,1,1],
    inertia=[0,0,0],
    actuated_joint=0,
    gravity_accel=1
  )
  q = sy.symbols('q_(1:4)', real=True)
  dq = sy.symbols('dq_(1:4)', real=True)
  M = get_kinetic_energy_mat(par, q)
  sy.pprint(M)
  C = get_coriolis_mat(M, q, dq)
  sy.pprint(C)

  U = get_potential_energy(par, q)
  sy.pprint(U)
  G = get_gravity_mat(U, q)
  sy.pprint(G)

if __name__ == "__main__":
  # test()
  show_equations()
