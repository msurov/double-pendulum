import sympy as sy


def jtimes(f, x, g):
  return f.jacobian(x) @ g

def lie(f, g, x):
  return jtimes(g, x, f) - jtimes(f, x, g)

def ad(f, g, k, x):
  result = g
  for i in range(k):
    result = lie(f, result, x)
  return result

def horzcat(*args):
  ny,_ = args[0].shape
  nx = 0

  for col in args:
    nx += col.shape[1]
    assert col.shape[0] == ny

  result = sy.zeros(ny, nx)
  i = 0

  for col in args:
    n = col.shape[1]
    result[:,i:i+n] = col
    i += n
  
  return result

def test():
  x, z, psi = sy.symbols('x z psi', real=True)
  g1 = sy.Matrix([
    sy.cos(psi), sy.sin(psi), 0
  ])
  g2 = sy.Matrix([
    0, 0, 1
  ])
  state = sy.Matrix([x, z, psi])
  g1_g2 = lie(g1, g2, state)
  Q = horzcat(g1, g2, g1_g2)
  print(Q.det())

def main():
  x, z, psi = sy.symbols('x z psi', real=True)
  dx, dz, dpsi = sy.symbols(R'\dot{x} \dot{z} \dot{\psi}', real=True)

  state = sy.Matrix([
    x, z, psi, dx, dz, dpsi
  ])

  f = sy.Matrix([
    dx, dz, dpsi, 0, -1, 0
  ])
  g1 = sy.Matrix([
    0, 0, 0, -sy.sin(psi), sy.cos(psi), 0
  ])
  g2 = sy.Matrix([
    0, 0, 0, 0, 0, 1
  ])

  elems = [
    f,
    g1,
    g2,
    ad(f, g1, 1, state),
    ad(f, g2, 1, state),
    ad(f, g1, 2, state),
  ]

  Q = horzcat(*elems)

  det_Q = Q.det()
  det_Q = sy.simplify(det_Q)s

  # sy.pprint(det_Q)
  print(sy.latex(det_Q))
  # print(sy.latex(Q))

main()
