from tora.dynamics import TORADynamics, TORAPar, tora_param_default
from collections import namedtuple
from common.numpy_utils import map_array, find_all_roots
from common.plots import set_pi_xticks
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca


def make_named(struct_name : str, **kwargs):
  T = namedtuple(struct_name, kwargs.keys())
  return T(**kwargs)

def get_reduced_dynamics2(X : ca.MX, arg : ca.MX, par : TORAPar):
  dX = ca.jacobian(X, arg)
  ddX = ca.jacobian(dX, arg)

  M = par.cart_mass + par.pendulum_mass
  m = par.pendulum_mass
  l = par.pendulum_length
  J = par.pendulum_inertia

  alpha = M * dX + m * l * ca.cos(arg)
  dalpha = ca.jacobian(alpha, arg)
  dalpha = ca.Function('dalpha', [arg], [dalpha])
  alpha = ca.Function('alpha', [arg], [alpha])
  beta = M * ddX - m * l * ca.sin(arg)
  beta = ca.Function('beta', [arg], [beta])
  kappa = par.damping_coef * dX
  kappa = ca.Function('kappa', [arg], [kappa])
  gamma = par.stiffness_coef * X
  gamma = ca.Function('gamma', [arg], [gamma])

  return make_named('ReducedDynamics', 
                    alpha = alpha, 
                    beta = beta, 
                    kappa = kappa, 
                    gamma = gamma,
                    dalpha = dalpha
                  )

def get_reduced_dynamics(X : ca.MX, arg : ca.MX, dynamics : TORADynamics):
  dX = ca.jacobian(X, arg)
  ddX = ca.jacobian(dX, arg)

  Q = ca.vertcat(X, arg)
  dQ = ca.vertcat(dX, 1)
  ddQ = ca.vertcat(ddX, 0)

  alpha = dynamics.B_perp_expr @ dynamics.M_expr @ dQ
  alpha = ca.substitute(alpha, dynamics.q, Q)

  dalpha = ca.jacobian(alpha, arg)
  dalpha = ca.Function('dalpha', [arg], [dalpha])

  alpha = ca.Function('alpha', [arg], [alpha])

  beta = dynamics.B_perp_expr @ (dynamics.M_expr @ ddQ + dynamics.C_expr @ dQ)
  beta = ca.substitute(beta, dynamics.q, Q)
  beta = ca.substitute(beta, dynamics.dq, dQ)
  beta = ca.Function('beta', [arg], [beta])

  kappa = dynamics.B_perp_expr @ dynamics.D_expr @ dQ
  kappa = ca.substitute(kappa, dynamics.q, Q)
  kappa = ca.Function('kappa', [arg], [kappa])

  gamma = dynamics.B_perp_expr @ dynamics.G_expr
  gamma = ca.substitute(gamma, dynamics.q, Q)
  gamma = ca.Function('gamma', [arg], [gamma])

  return make_named('ReducedDynamics', 
                    alpha = alpha, 
                    beta = beta, 
                    kappa = kappa, 
                    gamma = gamma,
                    dalpha = dalpha
                  )

def identify_singularities(reduced, interval):
  roots = find_all_roots(reduced.alpha, interval)
  singularities = []
  for r in roots:
    a = reduced.beta(r)
    b = reduced.kappa(r)
    c = reduced.gamma(r)
    d = float(b**2 - 4 * a * c)
    if d < 0:
      singularities.append((r, None, None))
    elif d > 0:
      v1 = float((-b + np.sqrt(d)) / (2 * a))
      v2 = float((-b - np.sqrt(d)) / (2 * a))
      v1, v2 = sorted([v1, v2])
      singularities.append((r, v1, v2))
    else:
      singularities.append((r, v1, v2))
      v = float(-b / (2 * a))
      singularities.append((r, v, None))
  return singularities

def speed_at_singularity(reduced, theta_s):
  a = reduced.beta(theta_s)
  b = reduced.kappa(theta_s)
  c = reduced.gamma(theta_s)
  d = float(b**2 - 4 * a * c)
  v1 = float((-b + np.sqrt(d)) / (2 * a))
  v2 = float((-b - np.sqrt(d)) / (2 * a))
  v1, v2 = sorted([v1, v2])
  return v1, v2

def integrate(reduced, theta0, dtheta0, interval, direction=1):
  def rhs(s, state):
    t, theta, dtheta = state
    Dt = reduced.alpha(theta)
    Dtheta = dtheta * Dt
    Ddtheta = -(reduced.beta(theta) * dtheta**2 + reduced.kappa(theta) * dtheta + reduced.gamma(theta))
    return [float(Dt), float(Dtheta), float(Ddtheta)]
  
  def stop_cond(s, state):
    t, *_ = state
    if t >= interval:
      return -1
    return 1

  stop_cond.terminal = True
  sol = solve_ivp(rhs, [0., direction * 100.], [0., theta0, dtheta0], max_step=1e-3, events=stop_cond)
  return sol.y[0], sol.y[1], sol.y[2]

def main():
  par = tora_param_default
  dynamics = TORADynamics(tora_param_default)
  theta = ca.MX.sym('theta')

  b = -0.1298696517944337
  theta_s = 1.
  a = -par.pendulum_mass * par.pendulum_length / (par.pendulum_mass + par.cart_mass)
  eps = 1e-3
  print(a)

  if False:
    step = 0.05  
    for i in range(20):
      X = a * ca.sin(theta) + b * ca.cos(theta - theta_s)
      reduced = get_reduced_dynamics(X, theta, dynamics)
      _,dtheta0 = speed_at_singularity(reduced, theta_s)
      t_, theta_, dtheta_ = integrate(reduced, theta_s + eps, dtheta0, 0.5, 1)
      print(f'dtheta0 = {dtheta0}, dthetaf = {dtheta_[-1]}, b = {b}')

      if abs(dtheta_[-1] - dtheta0) < 1e-3:
        break
      elif dtheta_[-1] < dtheta0:
        b += step
      else:
        b -= step
      step /= 2

  X = a * ca.sin(theta) + b * ca.cos(theta - theta_s)
  reduced = get_reduced_dynamics(X, theta, dynamics)
  singularities = identify_singularities(reduced, [-np.pi, np.pi])
  interval = 1.0

  if False:
    args = np.linspace(-np.pi, np.pi, 300)
    alpha = map_array(reduced.alpha, args, 1)
    beta = map_array(reduced.beta, args, 1)
    kappa = map_array(reduced.kappa, args, 1)
    gamma = map_array(reduced.gamma, args, 1)

    _, axes = plt.subplots(4, 1, sharex=True)
    axes[0].plot(args, alpha)
    axes[0].set_ylabel('alpha')
    axes[1].plot(args, beta)
    axes[1].set_ylabel('beta')
    axes[2].plot(args, kappa)
    axes[2].set_ylabel('kappa')
    axes[3].plot(args, gamma)
    axes[3].set_ylabel('gamma')

    for ax in axes:
      for s,_,__ in singularities:
        ax.axvline(s, color='red')
      ax.grid(True)

    s = singularities[0]
    beta = reduced.beta(s)
    gamma = reduced.gamma(s)
    kappa = reduced.kappa(s)
    dalpha = reduced.dalpha(s)
    print('beta', beta)
    print('gamma', gamma)
    print('kappa', kappa)
    d = kappa**2 - 4 * beta * gamma
    print('d', d)
    dtheta = (-kappa + np.sqrt(d)) / (2 * beta)
    print('dtheta', dtheta)
    lam = (-2 * beta * dtheta - kappa) / (dalpha * dtheta)
    print('lam', lam)

    set_pi_xticks('1/4')
    plt.tight_layout()
    plt.show()

  if True:
    for s,v1,v2 in singularities:
      print('singularity', s, 'speed', v1, 'and', v2)
      plt.axvline(s, color='red', ls='--')
      if v1 is not None:
        plt.plot(s, v1, 'o', color='red')
      if v2 is not None:
        plt.plot(s, v2, 'o', color='red')

    theta_s, dtheta_s,_ = singularities[1]
    t, theta, dtheta = integrate(reduced, theta_s - eps, dtheta_s, interval, -1)
    plt.plot(theta, dtheta, alpha=0.8, lw=1)

    t, theta, dtheta = integrate(reduced, theta_s + eps, dtheta_s, interval, -1)
    plt.plot(theta, dtheta, alpha=0.8, lw=1)

    theta_s,_,dtheta_s = singularities[1]
    t, theta, dtheta = integrate(reduced, theta_s - eps, dtheta_s, interval, 1)
    plt.plot(theta, dtheta, alpha=0.8, lw=1)

    t, theta, dtheta = integrate(reduced, theta_s + eps, dtheta_s, interval, 1)
    plt.plot(theta, dtheta, alpha=0.8, lw=1)

    theta_s,_,dtheta_s = singularities[0]
    t, theta, dtheta = integrate(reduced, theta_s + eps, dtheta_s, interval, -1)
    plt.plot(theta, dtheta, alpha=0.8, lw=1)

    t, theta, dtheta = integrate(reduced, theta_s - eps, dtheta_s, interval, -1)
    plt.plot(theta, dtheta, alpha=0.8, lw=1)

    theta_,dtheta_s,_ = singularities[0]
    t, theta, dtheta = integrate(reduced, theta_s + eps, dtheta_s, interval, 1)
    plt.plot(theta, dtheta, alpha=0.8, lw=1)

    t, theta, dtheta = integrate(reduced, theta_s - eps, dtheta_s, interval, 1)
    plt.plot(theta, dtheta, alpha=0.8, lw=1)

    plt.ylim(-30., 30.)
    plt.xlim(-1.2 * np.pi, 1.2 * np.pi)
    set_pi_xticks('1/4')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
  main()
