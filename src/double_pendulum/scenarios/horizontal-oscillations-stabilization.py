from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq
from transverse_dynamics.transverse_feedback import (
  TranverseFeedbackController,
  TransverseDynamics,
  TranverseFeedbackControllerPar
)
import numpy as np
import matplotlib.pyplot as plt
from common.linsys import (
  solve_gramian_mat,
  find_closed_loop_fund_mat,
)
from common.numpy_utils import (
  cont_angle,
  get_max_increasing_interval,
  map_array
)
from common.casadi_utils import ad
from common.plots import set_pi_xticks, set_pi_yticks
from double_pendulum.sim import (
  SimulationResult,
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
from visualization import (
  AnimGraph,
  AnimScope,
  Animate,
)
from double_pendulum.anim import (
  animate,
  DoublePendulumAnim,
  DoublePendulumViewParam,
  DoublePendulumView,
  get_view_parameters
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
from functools import reduce
from transverse_dynamics.cylindrical_transverse_coordinates import (
  CylindricalTransverseCoordinates,
  CylindricalTransverseCoordinatesPar,
)

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

def plot_transient(real_traj : Trajectory, ref_traj : Trajectory, q_sing : np.ndarray):
  fig, axes = plt.subplots(1, 3, sharex=True, num=f'Trajectory stabilization', figsize=(10, 3))

  ref_traj_par = dict(
    lw=2, ls='--', color='brown'
  )
  sim_traj_par = dict(
    lw=1, ls='-', color='#7070C0',
    alpha=0.8
  )
  sing_par = dict(
    color='grey',
    zorder=-1,
    lw=1,
    ls='--'
  )

  ax = axes[0]
  plt.sca(ax)
  plt.grid(True, ls='--', lw=1)
  plt.plot(real_traj.coords[:,0], real_traj.coords[:,1], **sim_traj_par)
  plt.plot(real_traj.coords[0,0], real_traj.coords[0,1], 'o', **sim_traj_par)
  n, = ref_traj.time.shape
  plt.plot(ref_traj.coords[:n//2,0], ref_traj.coords[:n//2,1], **ref_traj_par)
  plt.axvline(q_sing[0], **sing_par)
  set_pi_yticks('1/8')
  add_annotation(R'$q_1$', [0.45, -0.12])
  add_annotation(R'$q_2$', [-0.12, 0.55])
  plt.tick_params(direction='in')

  ax = axes[1]
  plt.sca(ax)
  plt.grid(True, ls='--', lw=1)
  plt.plot(real_traj.coords[:,0], real_traj.vels[:,0], **sim_traj_par)
  plt.plot(real_traj.coords[0,0], real_traj.vels[0,0], 'o', **sim_traj_par)
  plt.plot(ref_traj.coords[:,0], ref_traj.vels[:,0], **ref_traj_par)
  plt.axvline(q_sing[0], **sing_par)
  add_annotation(R'$q_1$', [0.45, -0.12])
  add_annotation(R'$\dot q_1$', [-0.12, 0.5])
  plt.yticks([-5, 0, 5, 10])
  plt.tick_params(direction='in')

  ax = axes[2]
  plt.sca(ax)
  plt.grid(True, ls='--', lw=1)
  plt.plot(real_traj.coords[:,0], real_traj.vels[:,1], **sim_traj_par)
  plt.plot(real_traj.coords[0,0], real_traj.vels[0,1], 'o', **sim_traj_par)
  plt.plot(ref_traj.coords[:,0], ref_traj.vels[:,1], **ref_traj_par)
  plt.axvline(q_sing[0], **sing_par)
  plt.yticks([-20, -10, 0, 10])
  plt.tick_params(direction='in')
  set_pi_xticks('1/12')
  add_annotation(R'$q_1$', [0.45, -0.12])
  add_annotation(R'$\dot q_2$', [-0.12, 0.5])

  plt.tight_layout(pad=1, h_pad=0)
  return fig

def plot_transverse_dynamics(usp : BSpline, sim_traj : Trajectory, controller_state : np.ndarray):
  ref_traj_par = dict(
    lw=2, ls='--', color='brown', alpha=0.8
  )
  sim_traj_par = dict(
    lw=1.5, ls='-', color='#7070C0',
    alpha=0.8
  )

  theta = controller_state[:,0]
  cont_angle(theta)
  xi = controller_state[:,1:4]
  w = controller_state[:,4]
  u = sim_traj.control
  V = controller_state[:,5]
  u_ref = usp(theta)

  fig, axes = plt.subplots(2, 1, sharex=True, num='transverse dynamics', figsize=(10, 3))

  plt.sca(axes[0])
  plt.grid(True, ls='--', color='grey')
  plt.plot(theta, 10 * xi[:,0], color='#6060C0', alpha=0.8, lw=1.5, label=R'$10 \cdot \xi_1$')
  plt.plot(theta, xi[:,1], color='#60C060', alpha=0.8, lw=1.5, label=R'$\xi_2$')
  plt.plot(theta, 100 * xi[:,2], color='#C06060', alpha=0.8, lw=1.5, label=R'$10^2 \cdot \xi_3$')
  plt.legend(ncols=3, fontsize=14, loc='upper right')
  plt.axhline(0, color='#606060', lw=1, zorder=-1)

  plt.sca(axes[1])
  plt.grid(True, ls='--', color='grey')
  plt.plot(theta, u, **sim_traj_par, label=R'$u(\tau)$')
  plt.plot(theta, u_ref, **ref_traj_par, label=R'$u_*(\tau)$')
  plt.legend(ncols=1, fontsize=14, loc='upper right')
  set_pi_xticks('1', fontsize=14)
  add_annotation(R'$\tau$', [0.46, -0.25])

  plt.xlim(np.pi - 0.1, 8*np.pi + 0.1)

  plt.tight_layout(pad=1, h_pad=0.1)
  return fig

def add_annotation(text : str, textpos : Tuple[float, float], fontsize=16):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.8
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': fontsize
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def make_anim(simres : SimulationResult, 
              ref_traj_of_theta : BSpline, 
              ref_ctrl_of_theta : BSpline,
              view_par : DoublePendulumViewParam, 
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
    DoublePendulumAnim(axes['anim'], view_par, traj_sim, shadow_color='#F0C0C0')
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

def show_phase_curve(reduced_dynamics : ReducedDynamics, x_diap, dx0):
  def rhs(x, st):
    dx, = st
    alpha = reduced_dynamics.alpha(x)
    beta = reduced_dynamics.beta(x)
    gamma = reduced_dynamics.gamma(x)
    return (-beta * dx**2 - gamma) / (alpha * dx)

  def stop_cond(x, st):
    dx, = st
    if dx <= 0:
      return -1
    return 1
  stop_cond.terminal = True

  sol = solve_ivp(rhs, x_diap, [dx0], max_step=1e-2, events=stop_cond)
  plt.plot(sol.t, sol.y[0], color='#8080C0', alpha=0.8, lw=1)
  plt.plot(sol.t, -sol.y[0], color='#8080C0', alpha=0.8, lw=1)

def show_phase_portrait(reduced_dynamics : ReducedDynamics, reduced_traj : Trajectory):
  eps = 1e-3
  fig, axes = plt.subplots(1, 2, figsize=(8,3), sharex=True)

  theta = np.linspace(-5, 1.5)
  alpha = map_array(reduced_dynamics.alpha, theta, 1)
  beta = map_array(reduced_dynamics.beta, theta, 1)
  gamma = map_array(reduced_dynamics.gamma, theta, 1)

  ax = axes[0]
  plt.sca(ax)
  plt.grid(True, lw=0.5, ls='--')
  plt.axhline(0, ls='-', lw=1, color='#404040')
  plt.axvline(0, ls='-', lw=1, color='#404040')

  plt.plot(theta, 1000 * alpha, label=R'$10^{3}\cdot\alpha(\theta)$', color='#6060C0')
  plt.plot(theta, 1000 * beta, label=R'$10^{3}\cdot\beta(\theta)$', color='#60C060')
  plt.plot(theta, gamma, label=R'$\gamma(\theta)$', color='#C06060')
  plt.legend()
  add_annotation(R'$\theta$', [0.48, -0.15])
  plt.tick_params(direction='in')

  ax = axes[1]
  plt.sca(ax)
  plt.grid(True, lw=0.5, ls='--')
  plt.axvline(0, ls='-', lw=1, color='#404040')
  plt.axhline(0, ls='-', lw=1, color='#404040')

  for x0 in np.linspace(-0.2, -4, 12, endpoint=False):
    show_phase_curve(reduced_dynamics, [x0, -eps], eps)

  for x0 in np.linspace(-0.2, -1.2, 10, endpoint=False):
    show_phase_curve(reduced_dynamics, [x0, -eps], 80)

  for x0 in np.linspace(-5, -4, 3, endpoint=False):
    show_phase_curve(reduced_dynamics, [x0, -eps], eps)

  for dx0 in np.linspace(30, 80, 4, endpoint=False):
    show_phase_curve(reduced_dynamics, [-5, -eps], dx0)

  for x0 in np.linspace(0.2, 1., 3, endpoint=False):
    show_phase_curve(reduced_dynamics, [x0, eps], eps)
    show_phase_curve(reduced_dynamics, [x0, eps], 80)

  for x0 in np.linspace(1.7, 1., 4):
    show_phase_curve(reduced_dynamics, [x0, eps], eps)
    show_phase_curve(reduced_dynamics, [x0, eps], 80)

  plt.plot(reduced_traj.coords, reduced_traj.vels, lw=2, color='#C04040')
  plt.xlim(-4.5, 1.5)
  plt.ylim(-80, 80)
  plt.yticks([-60, -30, 0, 30, 60])
  add_annotation(R'$\theta$', [0.5, -0.15])
  add_annotation(R'$\dot\theta$', [-0.08, 0.53])
  plt.tick_params(direction='in')

  plt.tight_layout(pad=1)

  return fig

def verify_reduced_dynamics(par : DoublePendulumParam, reduced : ReducedDynamics):
  par2 = convert_parameters(par)
  p1,p2,p3,p4,p5 = par2.p
  g = par2.gravity_accel
  qs = ca.DM([3*ca.pi/4, -ca.pi/4])
  k = -p2*p3**2*ca.sin(qs[1]) - 1/3*p2**2*p3*ca.sin(2*qs[1])

  theta = ca.SX.sym('theta')

  alpha = k*theta + p2*p3*ca.cos(qs[1]) - p2*p3*ca.cos(qs[1] + p3*theta + p2*theta*ca.cos(qs[1]) + k*theta**2/2/p3)
  beta = k + p2*p3**2*ca.sin(qs[1] + p3*theta + p2*theta*ca.cos(qs[1]) + k*theta**2/2/p3)
  gamma = -g*p5*ca.sin(qs[0] + qs[1] + p2*theta*ca.cos(qs[1]) + k*theta**2/2/p3)

  alpha_ = ca.evalf(ca.substitute(alpha, theta, 0.5))
  beta_ = ca.evalf(ca.substitute(beta, theta, 0.5))
  gamma_ = ca.evalf(ca.substitute(gamma, theta, 0.5))

  assert np.allclose(alpha_, -reduced.alpha(0.5))
  assert np.allclose(beta_, -reduced.beta(0.5))
  assert np.allclose(gamma_, -reduced.gamma(0.5))

def make_sample_traj(dynamics, par):
  constr_data = get_constr(par)
  constr = constr_data['constr']
  reduced = ReducedDynamics(dynamics, constr)
  # verify_reduced_dynamics(par, reduced)
  tr_1 = solve_reduced(reduced, [-4., -1e-3], 0.0, max_step=1e-3)
  tr_2 = solve_reduced(reduced, [1.0, 1e-3], 0.0, max_step=1e-3)[::-1]
  tr_reduced = reduce(traj_join, 
    [
      tr_1,
      tr_2,
      traj_reverse(tr_2),
      traj_reverse(tr_1),
    ]
  )

  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)
  theta_min = np.min(tr_reduced.coords)
  theta_max = np.max(tr_reduced.coords)

  return {
    **constr_data,
    'q_left': np.reshape(constr(theta_min), (2,)),
    'q_right': np.reshape(constr(theta_max), (2,)),
    'reduced_traj': tr_reduced,
    'traj': tr_orig,
    'reduced_dynamics': reduced
  }

class DoublePendulumFeedback(TranverseFeedbackController):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __call__(self, time, _signals, state):
    u = super().__call__(time, state)
    return u

def plot_tau(traj : Trajectory, real_traj : Trajectory, ctrl_state : np.ndarray, trans_coords : CylindricalTransverseCoordinates):
  period = traj.time[-1]

  tau = map_array(lambda p: trans_coords.forward_transform_fun(p)[0], traj.phase, 1)
  cont_angle(tau)
  curve_par = dict(
    color='#6060C0',
    lw=2,
  )

  fig, ax = plt.subplots(1, 1, figsize=(4,3), num='tau(t)')

  plt.sca(ax)
  plt.grid(True, ls='--', lw=0.5, color='#606060')
  plt.plot(traj.time, tau, **curve_par, label=R'$\tau(t)$')
  plt.plot(traj.time - period, tau - 2*np.pi, **curve_par)
  plt.plot(traj.time + period, tau + 2*np.pi, **curve_par)

  plt.xticks([-period/2, 0, period/2, period, 3*period/2],
    [R'$-\frac{T}{2}$', R'$0$', R'$\frac{T}{2}$', R'$T$', R'$\frac{3T}{2}$'],
    fontsize=14)
  set_pi_yticks('1', fontsize=14)
  plt.legend(fontsize=16)
  plt.xlim(-0.75 * period, 1.75 * period)
  plt.ylim(-1.6 * np.pi, 3.6 * np.pi)

  # plt.sca(axes[1])
  # plt.grid(True, ls='--', lw=0.5, color='#606060')
  # tau_actual = ctrl_state[:,0]
  # cont_angle(tau_actual)
  # t_actual = real_traj.time
  # plt.plot(t_actual, tau_actual)
  # for i in range(8):
  #   plt.plot(traj.time + i * period, tau + 2*i*np.pi, color='#C06000', lw=1, ls='--')
  # set_pi_yticks('2', fontsize=14)
  
  plt.tight_layout(pad=0.5)

  return fig

def plot_ltv(trans_dyn : TransverseDynamics, fb : TranverseFeedbackController):
  tau = np.linspace(-0.1, 2*np.pi + 0.1, 300)
  A = map_array(trans_dyn.A_fun, tau, (3,3))
  B = map_array(trans_dyn.B_fun, tau, (3,))
  K = map_array(fb.Ksp, tau, (3,))

  fig,axes = plt.subplots(5, 1, sharex=True, figsize=(8, 6), num='ltv')

  plt.sca(axes[0])
  plt.grid(True, ls='--', lw=1)
  plt.tick_params(direction='in')
  plt.plot(tau, 10 * A[:,0,0], label=R'$10 \cdot A_{1,1}$')
  plt.plot(tau, 10 * A[:,0,1], label=R'$10 \cdot A_{1,2}$')
  plt.plot(tau, A[:,0,2], label=R'$A_{1,3}$')
  plt.legend(ncols=3, fontsize=14)

  plt.sca(axes[1])
  plt.grid(True, ls='--', lw=1)
  plt.tick_params(direction='in')
  plt.plot(tau, 0.1 * A[:,1,0], label=R'$0.1 \cdot A_{2,1}$')
  plt.plot(tau, 10 * A[:,1,1], label=R'$A_{2,2}$')
  plt.plot(tau, 0.01 * A[:,1,2], label=R'$0.01 \cdot A_{2,3}$')
  plt.legend(ncols=3, fontsize=14)

  plt.sca(axes[2])
  plt.grid(True, ls='--', lw=1)
  plt.tick_params(direction='in')
  plt.plot(tau, 10 * A[:,2,0], label=R'$10 \cdot A_{3,1}$')
  plt.plot(tau, 100 * A[:,2,1], label=R'$100 \cdot A_{3,2}$')
  plt.plot(tau, A[:,2,2], label=R'$A_{3,3}$')
  plt.legend(ncols=3, fontsize=14)

  plt.sca(axes[3])
  plt.grid(True, ls='--', lw=1)
  plt.tick_params(direction='in')
  plt.plot(tau, 10 * B[:,0], label=R'$10 \cdot B_1$')
  plt.plot(tau, B[:,1], label=R'$B_2$')
  plt.plot(tau, 10 * B[:,2], label=R'$10 \cdot B_3$')
  plt.legend(ncols=3, fontsize=14)

  plt.sca(axes[4])
  plt.grid(True, ls='--', lw=1)
  plt.tick_params(direction='in')
  plt.plot(tau, 0.01 * K[:,0], label=R'$K_1$')
  plt.plot(tau, 0.1 * K[:,1], label=R'$K_2$')
  plt.plot(tau, 0.001 * K[:,2], label=R'$K_3$')
  plt.legend(ncols=3, fontsize=14)

  set_pi_xticks('1/2', fontsize=14)
  plt.xlim(-0.1, 2*np.pi + 0.1)

  add_annotation(R'$\tau$', [0.6, -0.2])

  plt.tight_layout(pad=0.5, h_pad=0.1)
  return fig

def plot_fb_gains(fb : TranverseFeedbackController):
  tau = np.linspace(-0.1, 2*np.pi + 0.1, 300)
  K = map_array(fb.Ksp, tau, (3,))

  fig,ax = plt.subplots(1, 1, sharex=True, figsize=(5, 3), num='ltv')

  plt.grid(True, ls='--', lw=1)
  plt.tick_params(direction='in')
  plt.plot(tau, 0.1 * K[:,0], label=R'$0.1 \cdot K_1$')
  plt.plot(tau, 1 * K[:,1], label=R'$K_2$')
  plt.plot(tau, 0.01 * K[:,2], label=R'$0.01 \cdot K_3$')
  plt.legend(fontsize=12)

  set_pi_xticks('1/2', fontsize=14)
  plt.xlim(-0.1, 2*np.pi + 0.1)

  add_annotation(R'$\tau$', [0.6, -0.12])

  plt.tight_layout(pad=0.5)
  return fig

def lie_algebra_dim(traj : Trajectory, dyn : DoublePendulumDynamics):
  fig, ax = plt.subplots(1, 1, num='Lie algebra det')
  x = ca.vertcat(dyn.q, dyn.dq)
  g0 = ca.substitute(dyn.rhs_expr, dyn.u, 0)
  tmp = ca.jacobian(dyn.rhs_expr, dyn.u)
  g1 = ca.substitute(tmp, dyn.u, 0)
  g2 = ad(g0, g1, 1, x)
  g3 = ad(g0, g2, 1, x)
  g4 = ad(g0, g3, 1, x)

  tmp = ca.horzcat(g0, g1, g2, g3)
  det1_expr = ca.det(tmp)
  det1 = ca.Function('det1', [x], [det1_expr])

  tmp = ca.horzcat(g0, g1, g2, g4)
  det2_expr = ca.det(tmp)
  det2 = ca.Function('det2', [x], [det2_expr])

  tmp = ca.fabs(det1_expr) + ca.fabs(det2_expr)
  det3 = ca.Function('det3', [x], [tmp])

  val3 = map_array(det3, traj.phase, 1)
  plt.figure('det psi')
  plt.plot(traj.time, val3)
  plt.grid(True)
  plt.tight_layout(pad=0.1)
  return fig

def make_feedback(q_central, traj, dynamics):
  trans_par = CylindricalTransverseCoordinatesPar(
    transverse_projection_mat = np.array([
      [0, 1, 0, 0], # 1/2
      [0, 0, 0, 1], # 1/50
    ]),
    proj_plane_x = np.array([1, 0, 0, 0]),
    proj_plane_y = np.array([0, 0, -1/25, 0]),
    proj_plane_origin = np.concatenate((q_central, [0, 0]))
  )

  coords = CylindricalTransverseCoordinates(traj, trans_par)
  trans_dyn = TransverseDynamics(dynamics, coords)

  t, W, F = solve_gramian_mat(trans_dyn.A_fun, trans_dyn.B_fun, [0, 2*np.pi])
  W_end = W[-1]
  W_evals = np.linalg.eigvals(W_end)
  print(f'controllability gramian eigvals: {W_evals[0]}, {W_evals[1]}, {W_evals[2]}')

  trans_fb_par = TranverseFeedbackControllerPar(
    # Q = np.diag([1/4, 1/2500., 1.]),
    Q = np.diag([5000., 10, 10000.]),
    R = np.eye(1),
    nsteps = 300
  )
  # fb = DoublePendulumTransverseFeedback(trans_dyn, trans_fb_par)
  fb = DoublePendulumFeedback(trans_dyn, trans_fb_par, max_step=5e-2, atol=1e-8, rtol=1e-8)

  t, F = find_closed_loop_fund_mat(trans_dyn.A_fun, trans_dyn.B_fun, fb.Ksp, [0, 2*np.pi], 
                                   max_step=1e-2, atol=1e-8, rtol=1e-8)
  F_2pi = F[-1]
  eigvals = np.linalg.eigvals(F_2pi)
  print(f'monodromy eigvals: {eigvals[0]}, {eigvals[1]}, {eigvals[2]}')

  return {
    'trans_coords': coords,
    'trans_dyn': trans_dyn,
    'feedback': fb,
  }

def motion_schematic(configurations, view_par : DoublePendulumViewParam):
  fig, ax = plt.subplots(1, 1)
  ax.set_aspect(1)
  plt.tick_params(direction='in')
  plt.grid(True, lw=0.5, ls='--', color='#404040', zorder=-1)

  print(configurations)

  for q in configurations:
    view = DoublePendulumView(view_par)
    view.move(q)
    for p in view.patches:
      ax.add_patch(p)

  # fig.savefig('fig/pendubot-horiz-oscillations-schem.pdf')
  plt.xlim(-0.1, 2)
  plt.ylim(-1.2, 0.2)
  plt.show()

def make_preview():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  data = make_sample_traj(dynamics, par)
  reduced_dynamics = data['reduced_dynamics']
  reduced_traj = data['reduced_traj']
  q_sing = data['singular_point']

  eps = 1e-3
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))

  theta = np.linspace(-5, 1.5)
  alpha = map_array(reduced_dynamics.alpha, theta, 1)
  beta = map_array(reduced_dynamics.beta, theta, 1)
  gamma = map_array(reduced_dynamics.gamma, theta, 1)

  theta_2 = brentq(reduced_dynamics.alpha, 1.8, 2.0)
  theta_2 = float(theta_2)
  dtheta_2 = np.sqrt(-reduced_dynamics.gamma(theta_2) / reduced_dynamics.beta(theta_2))
  dtheta_2 = float(dtheta_2)

  dtheta_1 = np.sqrt(-reduced_dynamics.gamma(0) / reduced_dynamics.beta(0))
  dtheta_1 = float(dtheta_1)

  ax = axes[0]
  plt.sca(ax)
  plt.grid(True, lw=0.5, ls='--')
  plt.axvline(0, ls='--', lw=1, color='#404040')
  plt.axhline(0, ls='--', lw=1, color='#404040')
  plt.axvline(theta_2, ls='--', lw=1, color='#404040')

  for x0 in np.linspace(-0.2, -4, 12, endpoint=False):
    show_phase_curve(reduced_dynamics, [x0, -eps], eps)

  for x0 in np.linspace(-0.2, -1.5, 10, endpoint=False):
    show_phase_curve(reduced_dynamics, [x0, -eps], 100)

  for x0 in np.linspace(-5, -4, 3, endpoint=False):
    show_phase_curve(reduced_dynamics, [x0, -eps], eps)

  for dx0 in np.linspace(30, 120, 8, endpoint=False):
    show_phase_curve(reduced_dynamics, [-5, -eps], dx0)

  for x0 in np.linspace(0.2, 1., 3, endpoint=False):
    show_phase_curve(reduced_dynamics, [x0, eps], eps)
    show_phase_curve(reduced_dynamics, [x0, eps], 100)

  for x0 in np.linspace(theta_2 - 0.1, 1., 4):
    show_phase_curve(reduced_dynamics, [x0, eps], eps)
    show_phase_curve(reduced_dynamics, [x0, eps], 100)

  for x0 in np.linspace(theta_2 + 0.1, 3.0, 8):
    show_phase_curve(reduced_dynamics, [x0, 3.0], 100)
    show_phase_curve(reduced_dynamics, [x0, 3.0], eps)

  show_phase_curve(reduced_dynamics, [theta_2 + eps, 3.0], dtheta_2)
  show_phase_curve(reduced_dynamics, [theta_2 - eps, eps], dtheta_2)

  plt.plot(reduced_traj.coords, reduced_traj.vels, lw=2, color='#C04040')
  plt.xlim(-4.5, 2.8)
  plt.ylim(-100, 100)
  plt.yticks([-60, -30, 0, 30, 60], [])
  plt.xticks(np.arange(-4, 3), [])
  add_annotation(R'$\theta$', [0.5, -0.15], fontsize=18)
  add_annotation(R'$\dot\theta$', [-0.08, 0.53], fontsize=18)
  plt.tick_params(direction='in')

  arrowprops = {
    'arrowstyle': "Simple, tail_width=0.05, head_width=0.5, head_length=0.7",
    'connectionstyle': "arc3,rad=0",
    'relpos': (1., 0.),
    'lw': 1.,
    'color': '#202020',
  }
  plt.annotate(
    '',
    xy = (0.0, -dtheta_1),
    xytext = (0.5, 0.05),
    xycoords = 'data',
    textcoords = 'axes fraction',
    arrowprops = arrowprops
  )
  plt.annotate(
    '',
    xy = (0.0, -dtheta_2),
    xytext = (0.5, 0.05),
    xycoords = 'data',
    textcoords = 'axes fraction',
    arrowprops = arrowprops
  )
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0.5,
    'alpha': 1
  }
  plt.annotate(
    'transition points',
    (0, 0),
    (0.4, 0.05),
    textcoords='axes fraction',
    xycoords='axes fraction',
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )

  # ax = axes[1]
  # plt.sca(ax)
  # ax.set_aspect(1)

  # view_par = DoublePendulumViewParam()
  # view = DoublePendulumView(view_par)
  # view.move(q_sing)

  # for p in view.patches:
  #   ax.add_patch(p)

  # plt.xlim(-0.2, 1.8)
  # plt.ylim(-1.2, 0.2)
  # plt.xticks([])
  # plt.yticks([])

  plt.tight_layout(pad=1, h_pad=0.1, w_pad=1)
  plt.show()

  return fig

def run_simulator():
  par = double_pendulum_param_default
  view_par = get_view_parameters(par)

  dynamics = DoublePendulumDynamics(par)
  data = make_sample_traj(dynamics, par)
  ref_traj = data['traj']
  constr = data['constr']
  q_sing = data['singular_point']
  q_left = data['q_left']
  q_right = data['q_right']

  if False:
    lie_algebra_dim(ref_traj, dynamics)

  if False:
    fig = motion_schematic([q_left, q_right], view_par)

  if False:
    reduced_dynamics = data['reduced_dynamics']
    reduced_traj = data['reduced_traj']
    fig = show_phase_portrait(reduced_dynamics, reduced_traj)
    fig.savefig('fig/pendubot-phase-portrait.pdf')

  ctrl_data = make_feedback(data['singular_point'], data['traj'], dynamics)
  fb = ctrl_data['feedback']

  trans_dyn = ctrl_data['trans_dyn']

  if False:
    plot_ltv(trans_dyn, fb)

  if True:
    fig = plot_fb_gains(fb)
    fig.savefig('fig/pendubot-feedback-gains.pdf')

  np.random.seed(0)
  simtime = 8 * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.zeros(4)
  x0[0] = 2.2
  x0[1] = 0.
  print('initial state:', x0)

  sim = DoublePendulumSimulator(double_pendulum_param_default, simulator_parameters_ideal, fb)

  simres = sim.run(x0, 0., simtime)
  res_traj = simres.traj
  trans_coords = ctrl_data['trans_coords']

  if True:
    fig = plot_tau(ref_traj, res_traj, simres.ctrl_state, trans_coords)
    fig.savefig('fig/pendubot-tau-of-time.pdf')

  if True:
    fig = plot_transient(res_traj, ref_traj, q_sing)
    fig.savefig('fig/pendubot-transient.pdf')
    fig = plot_transverse_dynamics(trans_coords.usp, simres.traj, simres.ctrl_state)
    fig.savefig('fig/pendubot-transverse.pdf')

  if True:
    a = make_anim(simres, trans_coords.xsp, trans_coords.usp, view_par, speedup=0.2)

  plt.show()

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  # integrate_closed_loop_system()
  # run_simulator()
  make_preview()
