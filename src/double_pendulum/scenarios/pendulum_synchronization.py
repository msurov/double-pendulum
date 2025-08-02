from double_pendulum.scenarios.sample_data import make_sample_data
from scipy.integrate import solve_ivp
from transverse_dynamics.transverse_feedback import (
  TransverseDynamics,
  TranverseFeedbackControllerPar
)
import numpy as np
import matplotlib.pyplot as plt
from common.trajectory import Trajectory
from common.numpy_utils import (
  cont_angle,
  get_max_increasing_interval,
  soft_sign
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
from double_pendulum.sim import SimulationResult
from double_pendulum.anim import (
  DoublePendulumAnim,
  DoublePendulumViewParam,
  animate,
  get_view_parameters
)
from visualization import (
  AnimGraph,
  AnimScope,
  Animate
)
from scipy.interpolate import BSpline
from typing import Tuple
from common.numpy_utils import map_array
from common.lqr import lqr_ltv, lqr_ltv_periodic
from scipy.interpolate import make_interp_spline


class TranverseFeedbackController:
  def __init__(self, reftraj : Trajectory, transdyn : TransverseDynamics, par : TranverseFeedbackControllerPar, **integ_args):
    self.coords_transform = transdyn.transverse_coords.forward_transform_fun
    self.u_ref = transdyn.transverse_coords.usp
    self.state = None
    self.__init_stab_coefs(transdyn, par)
    self.__init_bias(transdyn, par)
    self.__init_sync(reftraj)

  def __init_sync(self, reftraj : Trajectory):
    theta = map_array(lambda x: self.coords_transform(x)[0], reftraj.phase, 1)
    cont_angle(theta)
    _2_pi = 2 * np.pi
    period = reftraj.time[-1]
    theta0 = theta[0]
    time_sup = reftraj.time - period * (theta - theta0) / _2_pi
    self.time_sup = make_interp_spline(theta, time_sup, bc_type='periodic')
    self.traj_period = period
    self.theta0 = theta0

  def __ref_time(self, theta : float):
    _2_pi = 2 * np.pi
    return self.time_sup(theta) + self.traj_period * (theta - self.theta0) / _2_pi

  def __latency(self, cur_time : float, theta : float):
    ref_time = self.__ref_time(theta)
    period = self.traj_period
    half_period = period / 2
    latency = (cur_time - ref_time + half_period) % period - half_period
    return latency

  def __init_stab_coefs(self, transdyn : TransverseDynamics, par : TranverseFeedbackControllerPar):
    theta = np.linspace(0, 2*np.pi, 300)
    K, P = lqr_ltv_periodic(theta, transdyn.A_fun, transdyn.B_fun, par.Q, par.R)
    K_sp = make_interp_spline(theta, K, bc_type='periodic')
    P_sp = make_interp_spline(theta, P, bc_type='periodic')
    self.K_sp = K_sp
    self.P_sp = P_sp

  def __init_bias(self, transdyn : TransverseDynamics, par : TranverseFeedbackControllerPar):
    R_inv = np.linalg.inv(par.R)

    def sync_term_rhs(theta, eta):
      A = transdyn.A_fun(theta)
      B = transdyn.B_fun(theta)
      P = self.P_sp(theta)

      q = transdyn.At_fun(theta).T
      r = transdyn.Bt_fun(theta).T

      expr1 = (A.T - P @ B @ R_inv @ B.T)
      expr2 = P @ B @ R_inv @ r
      d_eta = -expr1 @ eta - expr2 - q
      d_eta = np.reshape(d_eta, (3,))
      return d_eta

    theta, eta = solve_periodic_iter(sync_term_rhs, [2*np.pi, 0], np.zeros(3), max_step=1e-2, maxiter=10)
    eta = eta[::-1]
    theta = theta[::-1]
    u_bias = np.zeros(theta.shape)

    for i in range(len(theta)):
      B = transdyn.B_fun(theta[i])
      r = transdyn.Bt_fun(theta[i])[0,0]
      u_bias[i] = -R_inv @ (B.T @ eta[i] + r)

    self.u_bias_sp = make_interp_spline(theta, u_bias, bc_type='periodic')

  def compute(self, t : float, x : np.ndarray) -> float:
    val = self.coords_transform(x)
    theta = float(val[0])
    xi = np.reshape(val[1:], (-1,))
    K = self.K_sp(theta)
    u_stab = K @ xi
    u_ref = self.u_ref(theta)
    dt = self.__latency(t, theta)
    u_bias = 3000. * soft_sign(dt, 4e-2) * self.u_bias_sp(theta)
    u = u_ref + u_stab + u_bias
    P = self.P_sp(theta)
    V = xi.T @ P @ xi / 2
    self.state = [theta, *xi, *u_stab, V, dt]
    return u

  def __call__(self, t, _, x):
    return self.compute(t, x)

def make_anim(simres : SimulationResult, 
              ref_traj : Trajectory,
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
    AnimGraph(axes['coords'], traj_sim.time, traj_sim.coords[:,0], alpha=0.8, lw=1),
    AnimGraph(axes['vels'], traj_sim.time, traj_sim.vels[:,0], alpha=0.8, lw=1),
    AnimGraph(axes['control'], traj_sim.time, traj_sim.control, alpha=0.8, lw=1),
    DoublePendulumAnim(axes['anim'], view_par, traj_sim, shadow_color='#F0C0C0')
  ]

  q_sp = make_interp_spline(ref_traj.time, ref_traj.coords, bc_type='periodic')
  dq_sp = make_interp_spline(ref_traj.time, ref_traj.vels, bc_type='periodic')
  ctrl_sp = make_interp_spline(ref_traj.time, ref_traj.control, bc_type='periodic')

  ax = axes['xi']
  plt.sca(ax)
  plt.grid(True)
  ax.set_title(R'transverse coordinates $\xi$')
  add_annotation(R'$t$', (0.92, 0.10))
  add_annotation(R'$\xi$', (0.03, 0.78))

  ax = axes['coords']
  plt.sca(ax)
  plt.plot(traj_sim.time, q_sp(traj_sim.time)[:,0], ls='-.', lw=1.5, color='red')
  plt.grid(True)
  add_annotation(R'$t$', (0.92, 0.10))
  add_annotation(R'$q_1$', (0.03, 0.78))

  ax = axes['vels']
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj_sim.time, dq_sp(traj_sim.time)[:,0], ls='-.', lw=1.5, color='red')
  add_annotation(R'$t$', (0.92, 0.10))
  add_annotation(R'$\dot q_1$', (0.03, 0.78))

  ax = axes['control']
  plt.sca(ax)
  plt.grid(True)
  ax.set_title(R'control input $u(\theta)$')
  plt.plot(traj_sim.time, ctrl_sp(traj_sim.time), ls='-.', lw=1.5, color='red')
  add_annotation(R'$t$', (0.92, 0.10))
  add_annotation(R'$u$', (0.03, 0.78))

  ax = axes['anim']
  ax.set_title(f'animation at {speedup:.2f}x speed')

  fig.tight_layout()

  return Animate(fig, animators, traj_sim.time[-1], fps, speedup, videopath)

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

def solve_periodic_iter(rhs, tspan, y0, maxiter=20, rtol=1e-6, atol=1e-6, **kwargs):
  for i in range(maxiter):
    sol = solve_ivp(rhs, tspan, y0, **kwargs)
    y_final = sol.y[:,-1]
    if np.allclose(y0, y_final, rtol, atol):
      break
    y0 = y_final
  sol.y[:,-1] = sol.y[:,0]
  return sol.t, sol.y.T

def main():
  data = make_sample_data()
  par = data['dynamics_par']
  dynamics = data['dynamics']
  transdyn = data['trans_dyn']
  ref_traj = data['traj']

  trans_par = TranverseFeedbackControllerPar(
    Q = np.diag([5., 2., 2.]),
    R = 1e-1 * np.eye(1),
    nsteps = 200
  )

  fb = TranverseFeedbackController(ref_traj, transdyn, trans_par)
  sim = DoublePendulumSimulator(double_pendulum_param_default, simulator_parameters_ideal, fb)

  np.random.seed(0)
  t0 = ref_traj.time[0]
  t1 = ref_traj.time[-1]
  simtime = 18 * (t1 - t0)
  step = 2e-3
  x0 = ref_traj.phase[0]
  x0 = x0 * (1 + 5e-2 * np.random.normal(size=x0.shape))
  res = sim.run(x0, 0.1, simtime)

  _, axes = plt.subplots(3, 1, num='phase')

  ax = axes[0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.coords[:,0], ref_traj.coords[:,1], ls='--', lw=2)
  plt.plot(res.traj.coords[:,0], res.traj.coords[:,1], lw=1, alpha=0.7)
  plt.ylabel(R'$q_2$')
  plt.xlabel(R'$q_1$')

  ax = axes[1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(ref_traj.coords[:,0], ref_traj.vels[:,0], ls='--', lw=1)
  plt.plot(res.traj.coords[:,0], res.traj.vels[:,0], lw=1, alpha=0.7)
  plt.ylabel(R'$\dot{q}_1$')
  plt.xlabel(R'$q_1$')

  dt = res.ctrl_state[:,6]

  ax = axes[2]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(res.traj.time, dt)
  plt.ylabel('latency, sec')

  view_par = get_view_parameters(par)
  a = make_anim(res, ref_traj, view_par, speedup=0.25, fps=60)
  a.anim.save('fig/anim.mp4', dpi=150, fps=60)
  plt.show()


if __name__ == '__main__':
  main()
