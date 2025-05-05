from pvtol.dynamics import PVTOLAircraftDynamics
import casadi as ca


def lie(f, g, x):
  return ca.jtimes(g, x, f) - ca.jtimes(f, x, g)

def ad(f, g, k, state):
  result = g
  for i in range(k):
    result = lie(f, result, state)
  return result

def main():
  dynamics = PVTOLAircraftDynamics()
  x = dynamics.phase
  f = dynamics.f_expr
  g1 = dynamics.g_expr[:,0]
  g2 = dynamics.g_expr[:,1]
  Q = ca.hcat([
    f,
    ad(f, g1, 0, x),
    ad(f, g2, 0, x),
    ad(f, g1, 1, x),
    ad(f, g2, 1, x),
    ad(f, g1, 2, x),
  ])
  det_Q = ca.det(Q)
  det_Q = ca.simplify(det_Q)
  print(det_Q)

main()
