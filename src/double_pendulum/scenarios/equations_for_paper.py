from double_pendulum.dynamics.dynamics_sympy_v2 import (
  DoublePendulumParam2,
  DoublePendulumDynamics
)
from double_pendulum.dynamics import dynamics_sympy
import sympy as sy


def test():
  p = sy.symbols('p_(1:6)', real=True, positive=True)
  g = sy.symbols('g', real=True, positive=True)
  par = DoublePendulumParam2(
    p = p,
    actuated_joint = 0,
    gravity_accel = g
  )
  dyn = DoublePendulumDynamics(par)

  q = dyn.q
  dq = dyn.dq
  M = dyn.M_expr
  C = dyn.C_expr
  G = dyn.G_expr
  B = dyn.B_expr
  J = sy.Matrix([
    [0, 1], 
    [-1, 0]
  ])
  B_perp = J @ B

  q_sing = (-sy.pi/4, 3*sy.pi/4)

  q1 = M.adjugate() @ B
  print("M' * B")
  sy.pprint(q1)

  q1 = q1.subs(zip(q, q_sing))

  k = sy.symbols('k', real=True)
  q2 = k * B_perp

  phi = sy.symbols('phi', real=True)
  constr = sy.Matrix(q_sing) + q1 * phi + q2 * phi**2 / 2
  Dconstr = constr.diff(phi)
  DDconstr = Dconstr.diff(phi)

  alpha = (B_perp.T @ M @ Dconstr)[0,0]
  alpha = alpha.subs(zip(q, constr))
  beta = (B_perp.T @ M @ DDconstr + B_perp.T @ C @ Dconstr)[0,0]
  beta = beta.subs(zip(q, constr))
  beta = beta.subs(zip(dq, Dconstr))
  gamma = (B_perp.T @ G)[0,0]
  gamma = gamma.subs(zip(q, constr))
  gamma = gamma.simplify()
  Dalpha = alpha.diff(phi)

  print('alpha(0):')
  sy.pprint(alpha.subs(phi, 0))
  print('Dalpha(0):')
  tmp = Dalpha.subs(phi, 0)
  tmp = tmp.simplify()
  print(sy.latex(tmp))
  print('beta(0):')
  sy.pprint(beta.subs(phi, 0))
  print('gamma:')
  print(sy.latex(gamma))

  print('beta/Dalpha:')
  tmp = beta / Dalpha
  tmp = tmp.subs(phi, 0).simplify()
  print(sy.latex(tmp))

  # expr = B_perp.T @ M @ q1
  # expr.simplify()
  # sy.pprint(expr)

  # sy.pprint(dyn.M_expr)
  # sy.pprint(dyn.B_expr)

test()
