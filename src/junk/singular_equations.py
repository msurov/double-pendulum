import scienceplots
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

cos = np.cos
sin = np.sin
sqrt = np.sqrt

def main():
  def alpha(theta):
    return (l - l * cos(theta) + a * theta * cos(theta))

  def beta(theta):
    return a * cos(theta)

  def gamma(theta):
    return -g * sin(theta)
  
  def kappa(theta):
    return 0.2
  
  def rhs(s, state):
    theta, omega = state
    dtheta = omega * alpha(theta)
    domega = -beta(theta) * omega**2 - kappa(theta) * omega - gamma(theta)
    return [dtheta, domega]

  def stop_cnd(s, state):
    theta, omega = state
    if abs(theta) > 1 or abs(omega) > 2:
      return -1
    return 1

  stop_cnd.terminal = True

  l = 1.0
  a = 0.3
  g = 1.0

  plot_par = dict(
    color='blue',
    lw=1.0,
    alpha=0.5
  )

  integ_par = dict(
    max_step=1e-2,
    events=stop_cnd
  )

  plt.figure('singularity-equilibrium')
  plt.grid(True)

  for theta0 in np.arange(0.05, 1.0, 0.1):
    sol = solve_ivp(rhs, [0, 100], [theta0, 0.0], **integ_par)
    plt.plot(sol.y[0], sol.y[1], **plot_par)

    sol = solve_ivp(rhs, [0, -100], [theta0, 0.0], **integ_par)
    plt.plot(sol.y[0], sol.y[1], **plot_par)

  for theta0 in np.arange(0.01, 1.0, 0.07):
    sol = solve_ivp(rhs, [0, 100], [theta0, 2.0], **integ_par)
    plt.plot(sol.y[0], sol.y[1], **plot_par)

    sol = solve_ivp(rhs, [0, -100], [theta0, -2.0], **integ_par)
    plt.plot(sol.y[0], sol.y[1], **plot_par)

  for theta0 in np.arange(-0.32, -0.01, 0.05):
    sol = solve_ivp(rhs, [0, 100], [theta0, 0.0], **integ_par)
    plt.plot(sol.y[0], sol.y[1], **plot_par)

    sol = solve_ivp(rhs, [0, -100], [theta0, 0.0], **integ_par)
    plt.plot(sol.y[0], sol.y[1], **plot_par)

  if True:
    sol = solve_ivp(rhs, [0, 10000], [0.001, 0.008], max_step=0.1)
    plt.plot(sol.y[0], sol.y[1], color='green', lw=2, alpha=0.8)

    sol = solve_ivp(rhs, [0, -10000], [0.001, -0.008], max_step=0.1)
    plt.plot(sol.y[0], sol.y[1], color='green', lw=2, alpha=0.8)

    dtheta0 = -kappa(0) / a
    sol = solve_ivp(rhs, [0, -10000], [0.001, dtheta0], max_step=0.1)
    plt.plot(sol.y[0], sol.y[1], **plot_par)

    sol = solve_ivp(rhs, [0, -100], [-0.001, dtheta0], max_step=0.1)
    plt.plot(sol.y[0], sol.y[1], **plot_par)

  plt.xlim(-0.2, 0.6)
  plt.ylim(-2, 2)
  plt.xlabel(R'$\theta$')
  plt.ylabel(R'$\dot\theta$')
  plt.tight_layout(pad=0.1)
  plt.savefig('fig/singularity-equilibrium/phase-with-friction.svg', dpi=600)
  plt.show()

if __name__ == '__main__':
  plt.style.use('science')
  plt.rcParams['legend.frameon'] = True
  plt.rcParams['legend.framealpha'] = 0.8
  main()
