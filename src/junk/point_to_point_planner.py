import sympy as sy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

x, theta = sy.symbols('x theta', real=True)
dx, dtheta = sy.symbols('dx dtheta', real=True)
M = sy.Matrix([
  [2, sy.cos(theta)],
  [sy.cos(theta), 1]
])
C = sy.Matrix([
  [0, -sy.sin(theta) * dtheta],
  [0, 0]
])
G = sy.Matrix([
  [0],
  [-sy.sin(theta)]
])
B = sy.Matrix([
  [1],
  [0]
])
B_perp = sy.Matrix([
  [0, 1]
])

V = M.adjugate() @ B
V = V.normalized()
N = B_perp.T

q0 = sy.Matrix([
  [0],
  [0]
])

V0 = V.subs({
  x: q0[0],
  theta: q0[1]
})
N0 = N.subs({
  x: q0[0],
  theta: q0[1]
})

eta = sy.symbols('eta', real=True)
Q = q0 + 1 * V0 * eta - 1 * N0 * eta**2 / 2
dQ = Q.diff(eta)
ddQ = dQ.diff(eta)

print('Q(0)', Q.subs(eta, 0).evalf())
print('Q(1)', Q.subs(eta, 1).evalf())
print('Q(-1)', Q.subs(eta, -1).evalf())
print('Q(-0.156)', Q.subs(eta, -0.156).evalf())

subs_vhc = {
  x: Q[0],
  theta: Q[1],
  dx: dQ[0],
  dtheta: dQ[1],
}

alpha = (B_perp @ M).subs(subs_vhc) @ dQ
alpha = alpha[0,0].simplify()
dalpha = alpha.diff(eta)

beta = (B_perp @ C).subs(subs_vhc) @ dQ + (B_perp @ M).subs(subs_vhc) @ ddQ
beta = beta[0,0].simplify()

gamma = (B_perp @ G).subs(subs_vhc)
gamma = gamma[0,0].simplify()
dgamma = gamma.diff(eta)

print('alpha:', alpha.subs(eta, 0))
print('beta:', beta.subs(eta, 0))
print('gamma:', gamma.subs(eta, 0))
print('dalpha:', dalpha.subs(eta, 0))
print('dgamma:', dgamma.subs(eta, 0))

alpha_fun = sy.lambdify(eta, alpha)
beta_fun = sy.lambdify(eta, beta)
gamma_fun = sy.lambdify(eta, gamma)

def rhs(x, st):
  dx, = st
  ddx = (-beta_fun(x) * dx**2 - gamma_fun(x)) / (alpha_fun(x) * dx)
  return [ddx]

def stop_cond(x, st):
  dx, = st
  if dx <= 0:
    return -1
  return 1

stop_cond.terminal = True

sol = solve_ivp(rhs, [1e-3, 1], [1e-3], max_step=1e-2, events=stop_cond)
eta = sol.t
deta = sol.y[0]
plt.plot(eta, deta, color='grey', lw=1)
plt.plot(eta, -deta, color='grey', lw=1)

sol = solve_ivp(rhs, [-1, -1e-2], [0.4], max_step=1e-2, events=stop_cond)
eta = sol.t
deta = sol.y[0]
plt.plot(eta, deta, color='grey', lw=1)
plt.plot(eta, -deta, color='grey', lw=1)

sol = solve_ivp(rhs, [-1e-3, -1], [1e-3], max_step=1e-2, events=stop_cond)
eta = sol.t
deta = sol.y[0]
plt.plot(eta, deta, color='grey', lw=1)
plt.plot(eta, -deta, color='grey', lw=1)

sol = solve_ivp(rhs, [-1, -1e-2], [0.5], max_step=1e-2, events=stop_cond)
eta = sol.t
deta = sol.y[0]
plt.plot(eta, deta, color='grey', lw=1)
plt.plot(eta, -deta, color='grey', lw=1)

sol = solve_ivp(rhs, [-0.5, -1e-2], [1e-3], max_step=1e-2, events=stop_cond)
eta = sol.t
deta = sol.y[0]
plt.plot(eta, deta, color='grey', lw=1)
plt.plot(eta, -deta, color='grey', lw=1)

sol = solve_ivp(rhs, [-0.7, -1e-2], [1e-3], max_step=1e-2, events=stop_cond)
eta = sol.t
deta = sol.y[0]
plt.plot(eta, deta, color='grey', lw=1)
plt.plot(eta, -deta, color='grey', lw=1)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid(True)
plt.show()
