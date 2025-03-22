import numpy as np
import sympy as sy
from double_pendulum.dynamics.parameters import DoublePendulumParam2
from typing import List, Tuple


sin = sy.sin
cos = sy.cos
Symbol = sy.Symbol
Matrix = sy.Matrix
pprint = sy.pprint

class DoublePendulumDynamics:
  def __init__(self, par : DoublePendulumParam2):
    q = sy.symbols(R'q_(1:3)', real=True)
    dq = sy.symbols(R'\dot{q}_(1:3)', real=True)
    u = sy.symbols('u', real=True)
    q1,q2 = q
    dq1,dq2 = dq
    p1,p2,p3,p4,p5 = par.p
    g = par.gravity_accel

    M = sy.Matrix([
      [p1 + 2 * p2 * cos(q2), p3 + p2 * cos(q2)],
      [p3 + p2 * cos(q2), p3]
    ])
    C = p2 * sin(q2) * sy.Matrix([
      [-2*dq2, -dq2],
      [dq1, 0]
    ])
    G = g * sy.Matrix([
      [-p4 * sin(q1) - p5 * sin(q1 + q2)],
      [-p5 * sin(q1 + q2)]
    ])
    B = sy.zeros(2, 1)
    B[par.actuated_joint] = 1

    self.u = u
    self.q = q
    self.dq = dq
    self.M_expr = M
    self.C_expr = C
    self.G_expr = G
    self.B_expr = B

    self.M = sy.lambdify([q], M)
    self.C = sy.lambdify([q, dq], C)
    self.G = sy.lambdify([q], G)
    self.B = sy.lambdify([q], B)

    ddq = M.inv() @ (-C @ sy.Matrix([[dq1], [dq2]]) - G + B * u)
    ddq.simplify()
    rhs = sy.zeros(4, 1)
    rhs[0:2,0] = dq
    rhs[2:4,0] = ddq
    self.rhs_expr = rhs
    state = (*q, *dq)
    self.rhs = sy.lambdify([state], rhs)

def get_sym_par() -> DoublePendulumParam2:
  p = sy.symbols('p_(1:6)', real=True, positive=True)
  g = sy.symbols('g', real=True, positive=True)

  return DoublePendulumParam2(
    p,
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
  print('parameters:')
  print('  first link length:', *d['lengths'])
  print('  links mass center displacements:', *d['mass_centers'])
  print('  links masses:', *d['masses'])
  print('  links inertia momentum:', *d['inertia'])
  print('  gravity acceleration:', d['gravity_accel'])
