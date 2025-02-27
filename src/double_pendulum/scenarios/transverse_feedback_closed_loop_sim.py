from double_pendulum.scenarios.sample_data import make_sample_data
from scipy.integrate import solve_ivp
from transverse_dynamics.transverse_feedback import (
  TranverseFeedbackController,
  TransverseDynamics,
  TranverseFeedbackControllerPar
)
import numpy as np
import matplotlib.pyplot as plt
from common.trajectory import Trajectory
from common.numpy_utils import cont_angle
from common.plots import set_pi_xticks, set_pi_yticks
from double_pendulum.sim import (
  DoublePendulumSimulator,
  DoublePendulumFeedback,
  DoublePendulumSignals,
  simulator_parameters_ideal,
  simulator_parameters_default,
)
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  double_pendulum_param_default
)
from double_pendulum.anim import animate


def show_trajectory(real_traj : Trajectory, ref_traj : Trajectory, savetofile=None):
  fig, axes = plt.subplots(2, 2, sharex=True, num=f'Trajectory stabilization')
  ax = axes[0,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.time, ref_traj.coords[:,1], lw=2, ls='--', color='green')
  plt.plot(real_traj.time, real_traj.coords[:,1], lw=1, color='lightblue')
  plt.ylabel(R'$q_2$')

  ax = axes[0,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.time, ref_traj.vels[:,0], lw=2, ls='--', color='green')
  plt.plot(real_traj.time, real_traj.vels[:,0], lw=1, color='lightblue')
  plt.ylabel(R'$\dot q_1$')

  ax = axes[1,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.time, ref_traj.vels[:,1], lw=2, ls='--', color='green')
  plt.plot(real_traj.time, real_traj.vels[:,1], lw=1, color='lightblue')
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\dot q_2$')

  ax = axes[1,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.time, ref_traj.control, lw=2, ls='--', color='green')
  plt.plot(real_traj.time, real_traj.control, lw=1, color='lightblue')
  plt.xlabel(R'$t$')
  plt.ylabel(R'$u$')

  plt.tight_layout()

  if savetofile is not None:
    plt.savefig(savetofile)

def show_projections(real_traj : Trajectory, ref_traj : Trajectory, savetofile=None):
  fig, axes = plt.subplots(2, 2, sharex=True, num=f'Trajectory stabilization')
  ax = axes[0,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(real_traj.coords[:,0], real_traj.coords[:,1], lw=1, color='blue', alpha=0.5)
  plt.plot(ref_traj.coords[:,0], ref_traj.coords[:,1], lw=2, ls='--', color='green')
  plt.ylabel(R'$q_2$')

  ax = axes[0,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(real_traj.coords[:,0], real_traj.vels[:,0], lw=1, color='blue', alpha=0.5)
  plt.plot(ref_traj.coords[:,0], ref_traj.vels[:,0], lw=2, ls='--', color='green')
  plt.ylabel(R'$\dot q_1$')

  ax = axes[1,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(real_traj.coords[:,0], real_traj.vels[:,1], lw=1, color='blue', alpha=0.5)
  plt.plot(ref_traj.coords[:,0], ref_traj.vels[:,1], lw=2, ls='--', color='green')
  plt.xlabel(R'$q_1$')
  plt.ylabel(R'$\dot q_2$')

  ax = axes[1,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(real_traj.coords[:,0], real_traj.control, lw=1, color='blue', alpha=0.5)
  plt.plot(ref_traj.coords[:,0], ref_traj.control, lw=2, ls='--', color='green')
  plt.xlabel(R'$q_1$')
  plt.ylabel(R'$u$')

  plt.tight_layout()

  if savetofile is not None:
    plt.savefig(savetofile)

def integrate_closed_loop_system():
  data = make_sample_data()
  transdyn = data['trans_dyn']
  dynamics = data['dynamics']
  traj = data['traj']

  par = TranverseFeedbackControllerPar(
    Q = np.diag([5., 5., 1.]),
    R = 1e-2 * np.eye(1),
    nsteps = 200
  )
  fb = TranverseFeedbackController(transdyn, par)

  def rhs(t, st):
    u = fb.compute(st)
    dst = dynamics.rhs(st, u)
    return np.reshape(dst, (-1,))
  
  t0 = traj.time[0]
  t1 = traj.time[-1]
  simtime = 3 * (t1 - t0)
  step = (t1 - t0) / 100
  x0 = traj.phase[0]
  np.random.seed(0)
  x0 = x0 * (1 + 1e-2 * np.random.normal(size=x0.shape))
  sol = solve_ivp(rhs, [0, simtime], x0, max_step=step)

  _, axes = plt.subplots(2, 1, sharex=True)

  ax = axes[0]
  plt.sca(ax)
  plt.plot(sol.y[0], sol.y[1], alpha=0.5)
  plt.plot(traj.coords[:,0], traj.coords[:,1], lw=2, ls='--')

  ax = axes[1]
  plt.sca(ax)
  plt.plot(sol.y[0], sol.y[2], alpha=0.5)
  plt.plot(traj.coords[:,0], traj.vels[:,0], lw=2, ls='--')

  plt.show()

class DoublePendulumTransverseFeedback(DoublePendulumFeedback):
  def __init__(self, transdyn : TransverseDynamics, par : TranverseFeedbackControllerPar):
    self.trans_feedback = TranverseFeedbackController(transdyn, par)
    self.state = None
  
  def __call__(self, time : float, sig : DoublePendulumSignals, state : np.ndarray):
    u, u_stab, theta, xi, V = self.trans_feedback.compute(state, full_output=True)
    self.state = np.array([*u_stab, theta, *xi, V])
    return u

def show_transverse_dynamics(controller_state : np.ndarray, saveto=None):
  u_stab = controller_state[:,0]
  theta = controller_state[:,1]
  cont_angle(theta)
  xi = controller_state[:,2:5]
  V = controller_state[:,5]

  _, axes = plt.subplots(2, 1, sharex=True, num='transverse dynamics')

  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(theta, xi)
  plt.legend([R'$\xi_1$', R'$\xi_2$', R'$\xi_3$'])

  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(theta, V)
  set_pi_xticks('1')
  plt.ylabel(R'Lyapunov function')
  plt.xlabel(R'$\theta$')
  plt.tight_layout()

  if saveto is not None:
    plt.savefig(saveto)

def run_simulator():
  data = make_sample_data()
  transdyn = data['trans_dyn']
  ref_traj = data['traj']

  # a = animate(ref_traj, data['dynamics_par'], speedup=0.25, videopath='data/stab.mp4')
  # plt.show()
  # exit()

  par = TranverseFeedbackControllerPar(
    Q = np.diag([5., 5., 1.]),
    R = 1e-2 * np.eye(1),
    nsteps = 200
  )
  fb = DoublePendulumTransverseFeedback(transdyn, par)

  simtime = 5. * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.array([-2., 0., 0, 0])

  sim = DoublePendulumSimulator(double_pendulum_param_default, simulator_parameters_ideal, fb)
  res = sim.run(x0, 0., simtime)
  show_transverse_dynamics(res.ctrl_state, 'data/transverse.png')
  show_projections(res.traj, ref_traj, 'data/projections.png')
  a = animate(res.traj, data['dynamics_par'], speedup=0.25, videopath='data/stab.mp4')
  plt.show()

if __name__ == "__main__":
  # integrate_closed_loop_system()
  run_simulator()
