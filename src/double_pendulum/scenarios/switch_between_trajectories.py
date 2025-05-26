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
  convert_parameters,
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
import casadi as ca
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  reconstruct_trajectory
)
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat,
  traj_reverse
)
from double_pendulum.anim import (
  draw,
  animate,
  animate_with_graphs,
  motion_schematic,
  motion_schematic_v2
)
from functools import reduce
from transverse_dynamics.cylindrical_transverse_coordinates import (
  CylindricalTransverseCoordinates,
  CylindricalTransverseCoordinatesPar,
)

def show_trajectory(real_traj : Trajectory, ref_traj : Trajectory):
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

def show_reference_trajectory(ref_traj : Trajectory):
  fig, axes = plt.subplots(2, 2, sharex=True, num=f'Trajectory stabilization')
  ax = axes[0,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.time, ref_traj.coords[:,1], lw=2, ls='--', color='green')
  plt.ylabel(R'$q_2$')

  ax = axes[0,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.time, ref_traj.vels[:,0], lw=2, ls='--', color='green')
  plt.ylabel(R'$\dot q_1$')

  ax = axes[1,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.time, ref_traj.vels[:,1], lw=2, ls='--', color='green')
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\dot q_2$')

  ax = axes[1,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.time, ref_traj.control, lw=2, ls='--', color='green')
  plt.xlabel(R'$t$')
  plt.ylabel(R'$u$')

  plt.tight_layout()

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
  theta_sim = controller_state[:,0]
  cont_angle(theta_sim)
  xi_sim = controller_state[:,1:4]
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

def get_constr(par : DoublePendulumParam):
  par2 = convert_parameters(par)
  p1,p2,p3,p4,p5 = par2.p
  g = par2.gravity_accel
  qs = ca.DM([3*ca.pi/4, -ca.pi/4])
  k = -p2*p3**2*ca.sin(qs[1]) - 1/3*p2**2*p3*ca.sin(2*qs[1])

  theta = ca.SX.sym('theta')
  q = qs + ca.vertcat(-p3, p3 + p2 * ca.cos(qs[1])) * theta + \
    k / (2 * p3) * ca.vertcat(0, 1) * theta**2
  
  q_fun = ca.Function('constr', [theta], [q])
  return {
    'constr': q_fun,
    'singular_point': np.reshape(qs, (2,))
  }

def make_sample_traj(dynamics, par):
  constr_data = get_constr(par)
  constr = constr_data['constr']
  reduced = ReducedDynamics(dynamics, constr)
  tr_1 = solve_reduced(reduced, [-5., -1e-3], 0.0, max_step=1e-3)
  tr_2 = solve_reduced(reduced, [1., 1e-3], 0.0, max_step=1e-3)[::-1]
  tr_3 = traj_reverse(tr_2)
  tr_5 = solve_reduced(reduced, [-3., -1e-3], 0.0, max_step=1e-3)
  tr_4 = traj_reverse(tr_5)
  tr_6 = solve_reduced(reduced, [1.5, 1e-3], 0.0, max_step=1e-3)[::-1]
  tr_7 = traj_reverse(tr_6)
  tr_8 = traj_reverse(tr_1)
  tr_reduced = reduce(traj_join, [tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8])
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)

  return {
    **constr_data,
    'traj_reduced': tr_reduced,
    'traj': tr_orig,
    'reduced_dynamics': reduced
  }

class DoublePendulumFeedback(TranverseFeedbackController):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __call__(self, time, _signals, state):
    u = super().__call__(time, state)
    return u

def make_feedback(q_central, traj, dynamics):
  trans_par = CylindricalTransverseCoordinatesPar(
    transverse_projection_mat = np.array([
      [0, 50, 0, 0],
      [0, 0, 0, 1],
    ]),
    proj_plane_x = np.array([50, 0, 0, 0]),
    proj_plane_y = np.array([0, 0, -1, 0]),
    proj_plane_origin = np.concatenate((q_central, [0, 0]))
  )

  coords = CylindricalTransverseCoordinates(traj, trans_par)
  trans_dyn = TransverseDynamics(dynamics, coords)

  trans_fb_par = TranverseFeedbackControllerPar(
    Q = np.diag([1., 1., 1.]),
    R = np.eye(1),
    nsteps = 300
  )
  # fb = DoublePendulumTransverseFeedback(trans_dyn, trans_fb_par)
  fb = DoublePendulumFeedback(trans_dyn, trans_fb_par, max_step=1e-2, atol=1e-8, rtol=1e-8)

  return {
    'trans_coords': coords,
    'trans_dyn': trans_dyn,
    'feedback': fb,
  }

def run_simulator():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  data = make_sample_traj(dynamics, par)
  ref_traj = data['traj']
  q_sing = data['singular_point']

  ctrl_data = make_feedback(data['singular_point'], data['traj'], dynamics)
  fb = ctrl_data['feedback']

  np.random.seed(0)
  simtime = 12 * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.zeros(4)
  x0[0] = 2.0
  x0[1] = 0.

  sim = DoublePendulumSimulator(double_pendulum_param_default, simulator_parameters_ideal, fb)

  simres = sim.run(x0, 0., simtime)
  res_traj = simres.traj

  trans_coords = ctrl_data['trans_coords']
  a = make_anim(simres, trans_coords.xsp, trans_coords.usp, par, speedup=0.2)
  plt.show()

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  # integrate_closed_loop_system()
  run_simulator()
