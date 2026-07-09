import scienceplots
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def a(y):
  return -2

def b(y):
  return -1

def c(y):
  return 3

def rhs(x, st):
  y, dy = st
  ddy = a(y) * dy**2 / y + b(y) * dy / y + c(y)
  return [dy, ddy]

def stop_condition(x, st):
  y, dy = st
  if abs(dy) > 3:
    return -1
  if y < -0.5 or y > 1.0:
    return -1
  if y**2 + dy**2 < 1e-8:
    return -1

  return 1

stop_condition.terminal = True

plt.style.use('science')
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8

plt.figure('phase portrait', figsize=(10, 6))
plt.grid(True)
plt.axvline(0, color='black', lw=1, ls='--')
plt.axhline(0, color='black', lw=1, ls='--')

for y0 in np.linspace(0.02, 0.5, 30):
  sol = solve_ivp(rhs, [0, 5], [y0, 0.0], max_step=1e-3, events=stop_condition)
  plt.plot(sol.y[0], sol.y[1], color='#9090F0', lw=1)

  sol = solve_ivp(rhs, [0, -5], [y0, 0.0], max_step=1e-3, events=stop_condition)
  plt.plot(sol.y[0], sol.y[1], color='#9090F0', lw=1)

for y0 in np.linspace(1e-2, 0.2, 12):
  sol = solve_ivp(rhs, [0, 5], [y0, 1.1], max_step=1e-3, events=stop_condition)
  plt.plot(sol.y[0], sol.y[1], color='#9090F0', lw=1)

  sol = solve_ivp(rhs, [0, -5], [y0, -1.1], max_step=1e-3, events=stop_condition)
  plt.plot(sol.y[0], sol.y[1], color='#9090F0', lw=1)

# trajectory originates from singularity
sol = solve_ivp(rhs, [0, 3], [1e-4, 0], max_step=1e-3, events=stop_condition)
plt.plot(sol.y[0], sol.y[1], color='#F09090', lw=2)

# traj passes through singularity
dy0_sing = -b(0) / a(0)
sol = solve_ivp(rhs, [0, -3], [1e-4, dy0_sing], max_step=1e-3, events=stop_condition)
plt.plot(sol.y[0], sol.y[1], color='#90F090', lw=2)
f1 = make_interp_spline(sol.y[0], sol.y[1], k=1)

sol = solve_ivp(rhs, [0, 3], [-1e-4, dy0_sing], max_step=1e-3, events=stop_condition)
plt.plot(sol.y[0], sol.y[1], color='#90F090', lw=2)
plt.fill_betweenx(sol.y[1], sol.y[0], 0, alpha=0.3, color='#90F090')

# traj comes to singularity
sol = solve_ivp(rhs, [0, -3], [1e-4, 0], max_step=1e-3, events=stop_condition)
plt.plot(sol.y[0], sol.y[1], color='#9090F0', lw=1)
f2 = make_interp_spline(sol.y[0], sol.y[1], k=1)

y = np.linspace(1e-4, 0.3, 1000)
dy1 = f1(y)
dy2 = f2(y)
plt.fill_between(y, dy1, dy2, alpha=0.3, color="#ECA053")

for y0 in np.linspace(-0.3, -0.02, 16):
  sol = solve_ivp(rhs, [0, 5], [y0, 0.0], max_step=1e-3, events=stop_condition)
  plt.plot(sol.y[0], sol.y[1], color='#9090F0', lw=1)

  sol = solve_ivp(rhs, [0, -5], [y0, 0.0], max_step=1e-3, events=stop_condition)
  print(sol.t[-1])
  plt.plot(sol.y[0], sol.y[1], color='#9090F0', lw=1)

plt.xlim(-0.2, 0.2)
plt.ylim(-1, 1.0)
plt.xlabel('$y$', fontsize=24)
plt.ylabel('$y\'$', fontsize=24)
plt.tight_layout()
plt.savefig('fig/singularity-equilibrium/phase-with-friction.png', dpi=300)
plt.show()
