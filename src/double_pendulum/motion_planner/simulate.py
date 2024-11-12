from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  Trajectory
)
from typing import Callable
from double_pendulum.anim import animate


def get_test_par() -> DoublePendulumParam:
  return DoublePendulumParam(
    lengths=[1., 1.],
    mass_centers=[0.5, 0.5],
    masses=[0.2, 0.2],
    inertia=[0.02, 0.02],
    actiated_joint=0,
    gravity_accel=9.81
  )

def integrate_mechsys(
      dyn : DoublePendulumDynamics,
      q0 : np.ndarray,
      dq0 : np.ndarray,
      t : np.ndarray,
      ufun : Callable[[float, np.ndarray], float],
      **integ_args
    ) -> Trajectory:
  def rhs(t, st):
    u = ufun(t, st)
    dst = dyn.rhs(st, u)
    return np.reshape(dst, (-1,))
  
  st0 = np.concatenate((q0, dq0))
  sol = solve_ivp(rhs, [t[0], t[-1]], st0, t_eval=t, **integ_args)
  nq, = np.shape(q0)
  state = sol.y.T
  u = np.array([ufun(ti,qi) for ti, qi in zip(t, state)], float)
  traj = Trajectory(
    time = t,
    phase = state,
    control = u
  )
  return traj

def main():
  par = get_test_par()
  dyn = DoublePendulumDynamics(par)  
  q0 = [0.0, 0.0]
  dq0 = [0.0, 0.0]
  st0 = [*q0, *dq0]

  ufun = lambda _, state: 0.2
  t = np.arange(0, 0.01, 0.0001)
  traj = integrate_mechsys(dyn, q0, dq0, t, ufun, max_step=1e-6)

  I1,I2 = par.inertia
  c1,c2 = par.mass_centers
  m1,m2 = par.masses
  l1,l2 = par.lengths

  p1 = I1 + I2 + c1**2 * m1 + c2**2 * m2 + l1**2 * m2
  p2 = m2 * c2 * l1
  p3 = I2 + m2 * c2**2
  p4 = c1 * m1 + l1 * m2
  p5 = c2 * m2

  Q = np.polyfit(t[0:3], traj.coords[0:3,:], 2)
  dQ = np.polyder(Q, 1)
  ddQ = dQ[0:1,:]

  Bperp = np.array([[0., 1.]])
  M = dyn.M(q0)
  C = dyn.C(q0, np.polyval(dQ, 0.))
  G = dyn.G(q0)
  alpha = Bperp @ M @ np.polyval(dQ, 0.)
  beta = Bperp @ M @ np.polyval(ddQ, 0.) + Bperp @ C @ np.polyval(dQ, 0.)
  gamma = Bperp @ G
  print(alpha)
  print(beta)
  print(gamma)

main()
