import sympy as sy
from double_pendulum.dynamics.dynamics_sympy_v2 import DoublePendulumDynamics, get_symbolic_dynamics, get_sym_par


def test():
  par = get_sym_par()

  g = par.gravity_accel
  p1,p2,p3,p4,p5 = par.p

  dyn = DoublePendulumDynamics(par)
  qs1 = sy.Symbol(R'q_{s,1}', real=True)
  qs2 = sy.Symbol(R'q_{s,2}', real=True)
  qs = (qs1, qs2)
  B_perp = sy.Matrix([[0, 1]])

  theta = sy.Symbol('theta', real=True)
  k = sy.Symbol('k', real=True)
  N = sy.Matrix([
    [-p3],
    [p3 + p2 * sy.cos(qs2)]
  ])
  F = sy.Matrix([[0], [1]]) / p3
  Phi = sy.Matrix([[qs1], [qs2]]) + N * theta + k * F / 2 * theta**2
  dPhi = sy.diff(Phi, theta)
  ddPhi = sy.diff(dPhi, theta)

  q_subs = {
    dyn.q[0]: Phi[0],
    dyn.q[1]: Phi[1],
  }
  dq_subs = {
    dyn.dq[0]: dPhi[0],
    dyn.dq[1]: dPhi[1],
  }

  alpha = B_perp @ dyn.M_expr.subs(q_subs) @ dPhi
  alpha = alpha[0,0].simplify()
  dalpha = sy.diff(alpha, theta)
  dalpha = dalpha.simplify()
  beta = B_perp @ dyn.M_expr.subs(q_subs) @ ddPhi + B_perp @ dyn.C_expr.subs(dq_subs).subs(q_subs) @ dPhi
  beta = beta[0,0].simplify()
  gamma = B_perp @ dyn.G_expr.subs(q_subs)
  gamma = gamma[0,0].simplify()

  print(R'\begin{align*}')
  print(R'\alpha\left(\theta\right) &=', sy.latex(alpha), R'\\')
  print(R'\beta\left(\theta\right) &=', sy.latex(beta), R'\\')
  print(R'\gamma\left(\theta\right) &=', sy.latex(gamma))
  print(R'\end{align*}')
  print()

  dalpha0 = dalpha.subs(theta, 0).simplify()
  beta0 = beta.subs(theta, 0).simplify()
  gamma0 = gamma.subs(theta, 0).simplify()

  print(R'\begin{align*}')
  print(R"\alpha'\left(0\right) &=", sy.latex(dalpha0), R'\\')
  print(R'\beta\left(0\right) &=', sy.latex(beta0), R'\\')
  print(R'\gamma\left(0\right) &=', sy.latex(gamma0))
  print(R'\end{align*}')

if __name__ == '__main__':
  test()
