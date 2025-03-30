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
from common.numpy_utils import (
  cont_angle,
  get_max_increasing_interval
)
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
  DoublePendulumParam,
  double_pendulum_param_default,
  double_pendulum_param_disturbed
)
from double_pendulum.anim import animate
from double_pendulum.sim import SimulationResult
from double_pendulum.anim import (
  AnimGraph,
  AnimScope,
  DoublePendulumAnim,
  Animate
)
from scipy.interpolate import BSpline
from typing import Tuple


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

def add_annotation(text : str, textpos : Tuple[float, float]):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.8
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': 18
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def make_anim(simres : SimulationResult, 
              ref_traj_of_theta : BSpline, 
              ref_ctrl_of_theta : BSpline,
              par : DoublePendulumParam, 
              fps=30., speedup=1.,
              videopath=None):

  controller_state = simres.ctrl_state
  w_sim = controller_state[:,0]
  theta_sim = controller_state[:,1]
  cont_angle(theta_sim)
  xi_sim = controller_state[:,2:5]
  traj_sim = simres.traj

  fig, axes = plt.subplot_mosaic([['xi', 'anim'],
                                  ['coords', 'anim'],
                                  ['vels', 'control']],
                                 figsize=(10, 7), layout='constrained',
                                #  dpi=300,
                                 num='animation')
  animators = [
    AnimGraph(axes['xi'], traj_sim.time, xi_sim),
    AnimScope(axes['coords'], traj_sim.time, traj_sim.coords[:,0], traj_sim.coords[:,1], alpha=0.6, lw=1.5),
    AnimScope(axes['vels'], traj_sim.time, traj_sim.coords[:,0], traj_sim.vels[:,0], alpha=0.6, lw=1.5),
    AnimScope(axes['control'], traj_sim.time, theta_sim, traj_sim.control, alpha=0.6, lw=1.5),
    DoublePendulumAnim(axes['anim'], par, traj_sim, shadow_color='#F0C0C0')
  ]

  ax = axes['xi']
  plt.sca(ax)
  plt.grid(True)
  ax.set_title(R'transverse coordinates $\xi$')
  add_annotation(R'$t$', (0.92, 0.10))
  add_annotation(R'$\xi$', (0.03, 0.78))

  ax = axes['coords']
  plt.sca(ax)
  ax.set_title(R'virtual holonomic constraint in $(q_1, q_2)$')
  theta = np.linspace(-np.pi, 0., 300)
  x_ref = ref_traj_of_theta(theta)
  q_ref = x_ref[:,0:2]
  plt.plot(q_ref[:,0], q_ref[:,1], ls='--', lw=2.5, color='red')
  plt.grid(True)
  add_annotation(R'$q_1$', (0.92, 0.10))
  add_annotation(R'$q_2$', (0.03, 0.78))

  ax = axes['vels']
  plt.sca(ax)
  plt.grid(True)
  ax.set_title(R'phase plane in $(q_1, \dot q_1)$')
  theta = np.linspace(-np.pi, np.pi, 300)
  x_ref = ref_traj_of_theta(theta)
  q_ref = x_ref[:,0:2]
  dq_ref = x_ref[:,2:4]
  plt.plot(q_ref[:,0], dq_ref[:,0], ls='--', lw=2.5, color='red')
  add_annotation(R'$q_1$', (0.92, 0.10))
  add_annotation(R'$\dot q_1$', (0.03, 0.78))

  ax = axes['control']
  plt.sca(ax)
  plt.grid(True)
  ax.set_title(R'control input $u(\theta)$')
  i1, i2 = get_max_increasing_interval(theta_sim)
  u_ref = ref_ctrl_of_theta(theta_sim)
  plt.plot(theta_sim[i1:i2], u_ref[i1:i2], ls='--', lw=2, color='red')
  add_annotation(R'$\theta$', (0.92, 0.10))
  add_annotation(R'$u$', (0.03, 0.78))
  set_pi_xticks('2')

  ax = axes['anim']
  ax.set_title(f'animation at {speedup:.2f}x speed')

  fig.tight_layout()

  return Animate(fig, animators, traj_sim.time[-1], fps, speedup, videopath)

def run_simulator():
  data = make_sample_data()
  par = data['dynamics_par']
  transdyn = data['trans_dyn']
  ref_traj = data['traj']

  trans_par = TranverseFeedbackControllerPar(
    Q = np.diag([5., 2., 2.]),
    R = 1e-2 * np.eye(1),
    nsteps = 200
  )
  fb = DoublePendulumTransverseFeedback(transdyn, trans_par)

  simtime = 12. * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.array([-1.0, 1.5, 0, 0])

  sim = DoublePendulumSimulator(double_pendulum_param_default, simulator_parameters_ideal, fb)
  res = sim.run(x0, 0., simtime)
  np.save('data/closed-loop-simulation-result.npy', res)
  show_transverse_dynamics(res.ctrl_state, 'data/transverse.png')
  show_projections(res.traj, ref_traj, 'data/projections.png')

  traj_of_theta = data['traj_of_theta']
  ctrl_of_theta = data['control_of_theta']
  a = make_anim(res, traj_of_theta, ctrl_of_theta, par, speedup=0.20, fps=60)
  # a.anim.save('fig/anim.mp4', dpi=300)
  plt.show()

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  # integrate_closed_loop_system()
  run_simulator()
