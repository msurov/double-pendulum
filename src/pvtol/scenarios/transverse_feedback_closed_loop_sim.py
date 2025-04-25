from scipy.integrate import solve_ivp
from transverse_dynamics.transverse_feedback import (
  TranverseFeedbackController,
  TranverseFeedbackControllerPar
)
from transverse_dynamics.transverse_coordinates import (
  TransverseCoordinates,
  TransverseDynamics
)
import numpy as np
import matplotlib.pyplot as plt
from common.trajectory import Trajectory
from common.numpy_utils import (
  cont_angle,
  get_max_increasing_interval
)
from common.plots import set_pi_xticks, set_pi_yticks
from scipy.interpolate import BSpline
from pvtol.dynamics import PVTOLAircraftDynamics
from typing import Tuple
from pvtol.scenarios.sample_data import make_sample_data
from common.lqr import lqr_ltv_periodic
from pvtol.sim import (
  PVTOLAircraftSimulator,
  PVTOLAircraftSimulatorPar,
  PVTOLAircraftFeedback,
  SimulationResult
)
from typing import Callable
from scipy.interpolate import make_interp_spline


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
  fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)

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
  set_pi_yticks('1/4')
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
  
def plot_transverse(simres : SimulationResult, uref : Callable):
  theta = simres.ctrl_state[:,0]
  npts = theta.shape[0] * 3 // 4
  theta = theta[:npts]
  xi = simres.ctrl_state[:npts,1:6]
  u_stab = simres.ctrl_state[:npts,6:8]
  V = simres.ctrl_state[:npts,8]
  u = simres.traj.control[:npts,:]

  fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7,5))
  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(theta, xi)
  plt.legend([Rf'$\rho_{i}$' for i in range(1, 7)], loc='lower right', ncol=3, fontsize=14)
  # add_annotation(R'$\rho$', [-0.08, 0.44])

  plt.sca(axes[1])
  plt.grid(True)
  set_pi_xticks('1/2')
  plt.plot(theta, u_stab)
  plt.legend([R'$w_1$', R'$w_2$'], loc='upper right', ncol=3, fontsize=14)
  # add_annotation(R'$w$', [-0.08, 0.40])

  plt.sca(axes[2])
  plt.grid(True)
  set_pi_xticks('1/2')
  axes[2].set_prop_cycle(None)
  plt.plot(theta, u)
  axes[2].set_prop_cycle(None)
  plt.plot(theta, uref(theta), '--', lw=2, alpha=0.6)
  plt.legend([R'$u_1$', R'$u_2$'], loc='upper right', ncol=3, fontsize=14)
  add_annotation(R'$\tau$', [0.55, -0.2])
  # add_annotation(R'$u$', [-0.08, 0.52])

  plt.tight_layout(h_pad=0, pad=0.5)
  plt.show()

def simulate_closed_loop_dynamics():
  data = make_sample_data()
  transdyn = data['transverse_dynamics']
  trans_coords = data['transverse_coordinates']
  ref_traj = data['traj']

  fb_par = TranverseFeedbackControllerPar(
    Q = np.eye(5),
    R = np.eye(2),
    nsteps = 1000,
    S = np.eye(5)
  )
  fb = TranverseFeedbackController(transdyn, fb_par)

  sim_par = PVTOLAircraftSimulatorPar(
    timestep = 1e-2,
    thrust_diap = [-10, 10],
    torque_diap = [-10, 10],
  )
  sim = PVTOLAircraftSimulator(sim_par, fb)

  simtime = 2. * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.array([0.1, -0.5, 0., 0., 0., 0.])
  simres = sim.run(x0, 0., simtime)

  show_transient(ref_traj, simres.traj, simres.ctrl_state, trans_coords.usp)

  plot_transverse(simres, trans_coords.usp)

  plt.show()

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  simulate_closed_loop_dynamics()
