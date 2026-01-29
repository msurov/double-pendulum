import scienceplots
from typing import Callable, Tuple
from cart_nlinks_pendulum.dynamics import (
  CartNLinksPendDynamics,
  CartNLinksPendPar, 
  LinkPar,
  cart_nlinks_pend_par_default
)
from cart_nlinks_pendulum.anim import get_vis_par, CartNLinksPendVis, launch_anim
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import common.mechsys_integ as ms
from common.trajectory import Trajectory


def rk4_step(f, x, h):
  k1 = f(x)
  k2 = f(x + 0.5*h*k1)
  k3 = f(x + 0.5*h*k2)
  k4 = f(x + h*k3)
  return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(sys : Callable[[np.ndarray], np.ndarray], x0 : np.ndarray, step : float, nsteps : int) -> Tuple[np.ndarray, np.ndarray]:
  x = x0
  t = 0.
  x_arr = [x]
  t_arr = [t]
  for i in range(1, nsteps + 1):
    x = rk4_step(sys, x, step)
    t += step
    x_arr.append(x)
    t_arr.append(t)
  return t_arr, x_arr

def traj1():
  par = CartNLinksPendPar(
    cart_mass = 1.,
    gravity_accel = 1.,
    links = 2 * [
      LinkPar(
        mass = 1,
        mass_center_displacement = 0.5,
        length = 1.
      )
    ]
  )
  sys = CartNLinksPendDynamics(par)
  n = par.ndof
  I = ca.DM.eye(n)
  B_perp = sys.B_perp_expr
  A1 = B_perp @ sys.M_expr @ I[:,0]
  A2 = B_perp @ sys.M_expr @ I[:,1:]
  fbcoef = 20.
  ddot_x = -fbcoef * sys.q[0]
  tmp = -A1 * ddot_x - B_perp @ (sys.C_expr @ sys.dq + sys.G_expr)
  tmp = ca.solve(A2, tmp)
  f = ca.vertcat(sys.dq, ddot_x, tmp)
  state = ca.vertcat(sys.q, sys.dq)
  rhs = ca.Function('f', [state], [f])

  dummy = ca.MX.sym('dummy')
  e1 = ca.solve(sys.M_expr, I[:,0])
  u_expr = (e1.T @ (sys.C_expr @ sys.dq + sys.G_expr) - fbcoef * sys.q[0]) / (e1.T @ sys.B_expr)
  u_fun = ca.Function('u', [dummy, sys.q, sys.dq], [u_expr])

  period = 2 * np.pi / np.sqrt(fbcoef)
  integtime = period / 2
  x0 = -0.5

  if True:
    q0 = np.array([x0, 1.84774741, 0.14646958])
    dq0 = np.zeros(3)

    traj = ms.integrate(sys, q0, dq0, integtime, u_fun, max_step=5e-3)
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)    
    ax1.plot(traj.coords[:,0], traj.vels[:,1])
    ax2.plot(traj.coords[:,0], traj.vels[:,2])

    traj = ms.integrate(sys, q0 + np.array([0.0, -0.1, 0.0]), dq0, integtime, u_fun, max_step=5e-3)
    ax1.plot(traj.coords[:,0], traj.vels[:,1])
    ax2.plot(traj.coords[:,0], traj.vels[:,2])

    traj = ms.integrate(sys, q0 + np.array([0.0, 0.0, -0.1]), dq0, integtime, u_fun, max_step=5e-3)
    ax1.plot(traj.coords[:,0], traj.vels[:,1])
    ax2.plot(traj.coords[:,0], traj.vels[:,2])

    ax1.set_ylabel('theta1')
    ax1.grid(True)
    ax2.set_ylabel('theta2')
    ax2.grid(True)

    plt.figure(figsize=(5,5))
    traj = ms.integrate(sys, q0, dq0, 2 * integtime, u_fun, max_step=5e-3)
    a = launch_anim(traj, par, speedup=0.2)
    plt.show()

  if False:
    nsteps = 300
    step = integtime / nsteps  

    dec_var = ca.MX.sym('dec_var', n - 1)
    dq0 = ca.DM.zeros(n)
    state0 = ca.vertcat(-0.5, dec_var, dq0)
    _, state = integrate(rhs, state0, step, nsteps)
    state1 = state[-1]
    dq1 = state1[n:]

    nlp = {
      'x': dec_var,
      'g': dq1,
      'f': 1
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    initial_guess = ca.DM([1.7, 0.1])
    solution = solver(x0=initial_guess, 
                      lbg=-1e-3,
                      ubg=1e-3, 
                      lbx=ca.DM([1.2, -0.2]),
                      ubx=ca.DM([2.5, 0.6]))
    x = tuple(solution['x'].full().flatten().tolist())
    # print(x)
    print('x0: [%.8f, %.8f]' % x)

def traj2():
  par = CartNLinksPendPar(
    cart_mass = 1.,
    gravity_accel = 1.,
    links = 2 * [
      LinkPar(
        mass = 1,
        mass_center_displacement = 0.5,
        length = 1.,
        inertia = 0.0
      )
    ]
  )
  sys = CartNLinksPendDynamics(par)
  n = par.ndof
  I = ca.DM.eye(n)
  B_perp = sys.B_perp_expr
  A1 = B_perp @ sys.M_expr @ I[:,0]
  A2 = B_perp @ sys.M_expr @ I[:,1:]
  fbcoef = 20.
  ddot_x = -fbcoef * sys.q[0]
  tmp = -A1 * ddot_x - B_perp @ (sys.C_expr @ sys.dq + sys.G_expr)
  tmp = ca.solve(A2, tmp)
  f = ca.vertcat(sys.dq, ddot_x, tmp)
  state = ca.vertcat(sys.q, sys.dq)
  rhs = ca.Function('f', [state], [f])

  dummy = ca.MX.sym('dummy')
  e1 = ca.solve(sys.M_expr, I[:,0])
  u_expr = (e1.T @ (sys.C_expr @ sys.dq + sys.G_expr) - fbcoef * sys.q[0]) / (e1.T @ sys.B_expr)
  u_fun = ca.Function('u', [dummy, sys.q, sys.dq], [u_expr])

  period = 2 * np.pi / np.sqrt(fbcoef)
  integtime = period / 2
  x0 = -0.5

  if False:
    nsteps = 300
    step = integtime / nsteps  

    dec_var = ca.MX.sym('dec_var', n - 1)
    dq0 = ca.DM.zeros(n)
    state0 = ca.vertcat(-0.5, dec_var, dq0)
    _, state = integrate(rhs, state0, step, nsteps)
    state1 = state[-1]
    dq1 = state1[n:]

    nlp = {
      'x': dec_var,
      'g': dq1,
      'f': 1
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    initial_guess = ca.DM([1.5, -0.5])
    solution = solver(x0=initial_guess, 
                      lbg=-1e-3,
                      ubg=1e-3, 
                      lbx=ca.DM([1.2, -2.0]),
                      ubx=ca.DM([2.5, 1.0]))
    x = tuple(solution['x'].full().flatten().tolist())
    # print(x)
    print('x0: [%.8f, %.8f]' % x)
    return

  if True:
    q0 = np.array([x0, 1.88407641, -0.55213895])
    dq0 = np.zeros(3)

    traj = ms.integrate(sys, q0, dq0, integtime, u_fun, max_step=5e-3)
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(traj.coords[:,0], traj.coords[:,1])
    ax2.plot(traj.coords[:,0], traj.coords[:,2])

    traj = ms.integrate(sys, q0 + np.array([0.0, -0.1, 0.0]), dq0, integtime, u_fun, max_step=5e-3)
    ax1.plot(traj.coords[:,0], traj.coords[:,1])
    ax2.plot(traj.coords[:,0], traj.coords[:,2])

    traj = ms.integrate(sys, q0 + np.array([0.0, 0.0, -0.1]), dq0, integtime, u_fun, max_step=5e-3)
    ax1.plot(traj.coords[:,0], traj.coords[:,1])
    ax2.plot(traj.coords[:,0], traj.coords[:,2])

    ax1.set_ylabel('theta1')
    ax1.grid(True)
    ax2.set_ylabel('theta2')
    ax2.grid(True)

    plt.figure(figsize=(4, 2))
    traj = ms.integrate(sys, q0, dq0, 6 * integtime, u_fun, max_step=5e-3)
    a = launch_anim(traj, par, speedup=0.5, fps=30)
    plt.tight_layout(pad=0)
    a.save('fig/cart-nlinks-pendulum/periodic-motion-anim.gif', dpi=500)
    plt.show()

if __name__ == '__main__':
  plt.style.use('science')
  plt.rcParams['legend.frameon'] = True
  plt.rcParams['legend.framealpha'] = 0.8
  traj2()
