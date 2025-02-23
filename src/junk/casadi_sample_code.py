import casadi as ca

def make_fun():
  a = ca.MX.sym('a')
  f = ca.Function('fun', [a], [ca.sin(a**2)])
  return f

a = ca.SX.sym('a')
f = make_fun()
f_expr = f(a)
Jf = ca.jacobian(f_expr, a)
print(Jf)