from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
from common.numpy_utils import cont_angle, map_array
from common.plots import set_pi_xticks, set_pi_yticks
from transverse_dynamics.transverse_coordinates import TransverseDynamics
from transverse_dynamics.transverse_feedback import TranverseFeedbackController
from pvtol.sim import SimulationResult


def add_annotation(text : str, textpos : Tuple[int, int]):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.2
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': 20
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def show_transient(ref_traj, sim_traj, ctrl_state, uref):
  fig, axes = plt.subplots(2, 2, figsize=(7, 4), sharex=True)

  theta = ctrl_state[:,0]
  cont_angle(theta)
  xi = ctrl_state[:,1:6]
  u_stab = ctrl_state[:,6:8]

  ref_curve_par = dict(
    ls='--', lw=4, color='#606060', alpha=0.6
  )
  sim_curve_par = dict(
    lw=0.8, color='blue'
  )

  plt.sca(axes[0,0])
  plt.grid(True)
  npts, = ref_traj.time.shape
  plt.plot(ref_traj.coords[:npts//2,0], ref_traj.coords[:npts//2,1], **ref_curve_par)
  plt.plot(sim_traj.coords[:,0], sim_traj.coords[:,1], **sim_curve_par)
  plt.plot(sim_traj.coords[0,0], sim_traj.coords[0,1], 'o', **sim_curve_par)
  # add_annotation(R'$x$', [0.58, -0.20])
  add_annotation(R'$z$', [-0.20, 0.58])
  plt.yticks([-0.4, -0.2, 0.])

  plt.sca(axes[0,1])
  plt.grid(True)
  npts, = ref_traj.time.shape
  plt.plot(ref_traj.coords[:npts//2,0], ref_traj.coords[:npts//2,2], **ref_curve_par)
  plt.plot(sim_traj.coords[:,0], sim_traj.coords[:,2], **sim_curve_par)
  plt.plot(sim_traj.coords[0,0], sim_traj.coords[0,2], 'o', **sim_curve_par)
  set_pi_yticks('1/4', fontsize=14)
  # add_annotation(R'$x$',    [0.58, -0.20])
  add_annotation(R'$\psi$', [-0.18, 0.58])

  plt.sca(axes[1,0])
  plt.grid(True)
  npts, = ref_traj.time.shape
  plt.plot(ref_traj.coords[:,0], ref_traj.vels[:,0], **ref_curve_par)
  plt.plot(sim_traj.coords[:,0], sim_traj.vels[:,0], **sim_curve_par)
  plt.plot(sim_traj.coords[0,0], sim_traj.vels[0,0], 'o', **sim_curve_par)
  add_annotation(R'$x$',      [0.58, -0.20])
  add_annotation(R'$\dot x$', [-0.20, 0.56])

  plt.sca(axes[1,1])
  plt.grid(True)
  plt.plot(ref_traj.coords[:,0], ref_traj.vels[:,2], **ref_curve_par)
  plt.plot(sim_traj.coords[:,0], sim_traj.vels[:,2], **sim_curve_par)
  plt.plot(sim_traj.coords[0,0], sim_traj.vels[0,2], 'o', **sim_curve_par)
  add_annotation(R'$x$',      [0.58, -0.20])
  add_annotation(R'$\dot \psi$', [-0.18, 0.56])

  plt.tight_layout(h_pad=0)
  return fig

def plot_linsys(transdyn : TransverseDynamics, fb : TranverseFeedbackController):
  fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7,4))
  theta_min = transdyn.transverse_coords.theta_min
  theta_max = transdyn.transverse_coords.theta_max
  theta = np.linspace(0, theta_max - theta_min, 400)
  A = map_array(transdyn.A_fun, theta)
  B = map_array(transdyn.B_fun, theta)

  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(theta, A[:,:,0])
  plt.plot(theta, A[:,:,1])
  plt.plot(theta, A[:,:,2])
  plt.plot(theta, A[:,:,3])
  plt.plot(theta, A[:,:,4])

  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(theta, B[:,:,0])
  plt.plot(theta, B[:,:,1])

  if fb is not None:
    K = map_array(fb.Ksp, theta)
    plt.sca(axes[2])
    plt.grid(True)
    plt.plot(theta, K[:,0,:])
    plt.plot(theta, K[:,1,:])

  set_pi_xticks('1/2', fontsize=14)
  plt.tight_layout(h_pad=0, pad=0.5)

  return fig

def plot_transverse(simres : SimulationResult, uref : Callable):
  theta = simres.ctrl_state[:,0]
  npts = theta.shape[0] * 3 // 4
  theta = theta[:npts]
  xi = simres.ctrl_state[:npts,1:6]
  u_stab = simres.ctrl_state[:npts,6:8]
  V = simres.ctrl_state[:npts,8]
  u = simres.traj.control[:npts,:]

  fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7,4))
  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(theta, xi[:,0], lw=1, label=R'$\rho_1$')
  plt.plot(theta, xi[:,1], lw=1, label=R'$\rho_2$')
  plt.plot(theta, xi[:,2], lw=1, label=R'$\rho_3$')
  plt.plot(theta, xi[:,3], lw=1, label=R'$\rho_4$')
  plt.plot(theta, xi[:,4], lw=1, label=R'$\rho_5$')
  plt.legend(ncol=3, fontsize=12)
  # add_annotation(R'$\rho$', [-0.08, 0.44])

  plt.sca(axes[1])
  plt.grid(True)
  set_pi_xticks('1/2', fontsize=14)
  plt.plot(theta, u_stab, lw=1)
  plt.legend([R'$w_1$', R'$w_2$'], loc='upper right', ncol=3, fontsize=12)
  # add_annotation(R'$w$', [-0.08, 0.40])

  plt.sca(axes[2])
  plt.grid(True)
  set_pi_xticks('1/2', fontsize=14)
  axes[2].set_prop_cycle(None)
  plt.plot(theta, u)
  axes[2].set_prop_cycle(None)
  plt.plot(theta, uref(theta), '--', lw=2, alpha=0.6)
  plt.legend([R'$u_1$', R'$u_2$'], loc='upper right', ncol=3, fontsize=12)
  add_annotation(R'$\tau$', [0.55, -0.25])
  # add_annotation(R'$u$', [-0.08, 0.52])

  plt.tight_layout(h_pad=0, pad=0.5)
  return fig
