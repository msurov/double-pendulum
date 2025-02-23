from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default
)
from double_pendulum.motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
from double_pendulum.motion_planner.singular_constrs import get_sing_constr_at
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def transverse_dynamics():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  singpt = [-2, 2.8]
  constr = get_sing_constr_at(dynamics, singpt)
  reduced = ReducedDynamics(dynamics, constr)

  y = ca.SX.sym('y', 2)
  dy = ca.SX.sym('dy', 2)
  N = ca.DM([[0], [1]])

  q = constr(y[0]) + N * y[1]
  J = ca.jacobian(q, y)
  J = ca.simplify(J)
  dq = J @ dy
  
  expr = ca.substitute(
    dynamics.ddq_expr,
    dynamics.q,
    q
  )
  expr = ca.substitute(
    expr,
    dynamics.dq,
    dq
  )
  ddq = ca.simplify(expr)
  ddy = ca.solve(J, ddq - ca.jtimes(J, y, dy) @ dy)
  g = ca.jacobian(ddy, dynamics.u)
  g = ca.substitute(g, dynamics.u, 0)
  g = ca.simplify(g)
  f = ca.substitute(ddy, dynamics.u, 0)
  f = ca.simplify(f)

  g_fun = ca.Function('g', [y], [g])
  
  s = np.linspace(-0.1, 0.1, 100)
  g = np.zeros(s.shape + (2,))
  f = np.zeros(s.shape + (2,))
  for i in range(s.shape[0]):
    g[i,:] = g_fun([s[i], 0.0]).T
  
  plt.plot(s, g)
  plt.grid(True)
  plt.show()

transverse_dynamics()
