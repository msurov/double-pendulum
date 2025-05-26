from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  double_pendulum_param_default
)
import casadi as ca
import numpy as np
from common.casadi_utils import ad, lie

def test_1():
  par = get_sym_par()
  dyn = DoublePendulumDynamics(par)
  rhs = dyn.rhs_expr

  g = rhs.diff(dyn.u)
  g.simplify()

  f = rhs.subs(dyn.u, 0)
  f.simplify()

  x = (*dyn.q, *dyn.dq)

  # sy.pprint(g)
  # sy.pprint(f)

  fg1 = ad(f, g, 1, x)
  fg1.simplify()
  fg2 = ad(f, fg1, 1, x)
  fg2.simplify()

  sy.pprint(f)
  sy.pprint(g)
  sy.pprint(fg1)
  sy.pprint(fg2)

  elems = [
    f,
    g,
    fg1,
    fg2,
  ]


def main():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  f = ca.substitute(dynamics.rhs_expr, dynamics.u, 0)
  g = ca.substitute(ca.jacobian(dynamics.rhs_expr, dynamics.u), dynamics.u, 0)

  x = ca.vertcat(dynamics.q, dynamics.dq)
  fg1 = lie(f, g, x)
  fg2 = lie(f, fg1, x)

  Q = ca.horzcat(f, g, fg1, fg2)
  det_Q = ca.det(Q)
  det_Q_fun = ca.Function('det_Q', [x], [det_Q])

  print(det_Q_fun([1,0,3,-4]))
  print(det_Q_fun([-1,0,-3,4]))
  
if __name__ == '__main__':
  main()
