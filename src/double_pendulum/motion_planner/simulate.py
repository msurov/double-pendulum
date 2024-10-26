from scipy.integrate import solve_ivp
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
    state = state,
    control = u
  )
  return traj

def main():
  par = get_test_par()
  dyn = DoublePendulumDynamics(par)  
  q0 = [0.0, 0.0]
  dq0 = [0., 0.]
  ufun = lambda _, state: 0.1
  t = np.arange(0, 1, 0.01)
  traj = integrate_mechsys(dyn, q0, dq0, t, ufun, max_step=1e-3)
  # a = animate(traj, par)
  # plt.show()
  q1,q2 = traj.coords.T
  plt.plot(q1, q2)
  plt.grid(True)
  plt.show()

main()
