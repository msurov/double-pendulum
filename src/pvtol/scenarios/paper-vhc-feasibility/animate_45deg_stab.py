from transverse_dynamics.transverse_feedback import (
  TranverseFeedbackController,
  TranverseFeedbackControllerPar
)
import numpy as np
import matplotlib.pyplot as plt
from common.trajectory import Trajectory
from common.plots import set_pi_xticks, set_pi_yticks
from common.numpy_utils import cont_angle, map_array
from pvtol.dynamics import PVTOLAircraftDynamics
from typing import Tuple
from pvtol.scenarios.sample_data import make_sample_data
from pvtol.sim import (
  PVTOLAircraftSimulator,
  PVTOLAircraftSimulatorPar,
  PVTOLAircraftFeedback,
  SimulationResult
)
from pvtol.anim import AnimPVTOLAircraft, AnimPVTOLAircraftPar
from pvtol.anim.draw import compute_occupancy_box, expand_box
from visualization.anim import Animate
from visualization.anim_graph import AnimScope, AnimTrace
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline


def add_annotation(text : str, textpos : Tuple[int, int], fontsize=22):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.9
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': fontsize
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def simulate_closed_loop_dynamics():
  data = make_sample_data('45deg')
  transdyn = data['transverse_dynamics']
  trans_coords = data['transverse_coordinates']
  ref_traj = data['traj']

  theta_min = np.min(data['reduced_traj'].coords)
  theta_max = np.max(data['reduced_traj'].coords)
  theta = np.linspace(theta_min, theta_max, 200)
  constr = data['constr']
  q_ref = map_array(constr, theta)

  fb_par = TranverseFeedbackControllerPar(
    Q = np.diag([12., 1., 8., 1., 1.]),
    R = 0.1 * np.eye(2),
    nsteps = 400,
    S = np.eye(5)
  )
  fb = TranverseFeedbackController(transdyn, fb_par)

  sim_par = PVTOLAircraftSimulatorPar(
    timestep = 1e-3,
    thrust_diap = [-20, 20],
    torque_diap = [-20, 20],
  )
  sim = PVTOLAircraftSimulator(sim_par, fb)

  simtime = 6. * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.array([0.01, -0.5, 0., 0., 0., 0.])
  simres = sim.run(x0, 0., simtime)

  fig = plt.figure('PVTOL anim', figsize=(9, 16))
  gs = gridspec.GridSpec(6, 2,
    width_ratios=[2, 1],
    height_ratios=[1, 1, 1, 0.75, 0.75, 0.75])

  # Add subplots
  ax1 = fig.add_subplot(gs[0:3, 0])
  ax2 = fig.add_subplot(gs[0, 1])
  ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
  ax4 = fig.add_subplot(gs[2, 1], sharex=ax2)
  ax5 = fig.add_subplot(gs[3, 0:2])
  ax6 = fig.add_subplot(gs[4, 0:2], sharex=ax5)
  ax7 = fig.add_subplot(gs[5, 0:2], sharex=ax5)

  animators = []

  plt.sca(ax1)
  ax1.set_aspect(1)
  plt.plot(q_ref[:,0], q_ref[:,1], color='#B0B0B0', lw=1)
  anim_par = AnimPVTOLAircraftPar(aircraft_size=0.6)
  anim_pvtol = AnimPVTOLAircraft(ax1, simres.traj, anim_par)
  animators.append(anim_pvtol)
  box = compute_occupancy_box(simres.traj.coords, anim_par.aircraft_size)
  xmin, xmax, ymin, ymax = expand_box(*box, 10)
  ax1.set_xlim(xmin, xmax)
  ax1.set_ylim(ymin, ymax)
  # ax1.set_xticks([])
  # ax1.set_yticks([])
  plt.axis('off')

  animators.append(AnimTrace(ax1, simres.traj.time, simres.traj.coords[:,0], simres.traj.coords[:,1], 1, False, alpha=0.5))

  plt.sca(ax2)
  plt.grid(True)
  add_annotation('$z$', [0.85, 0.78])
  plt.yticks([-0.4, -0.2, 0., 0.2])
  plt.plot(q_ref[:,0], q_ref[:,1], lw=3, color='red', ls='--', zorder=5)
  animators.append(AnimScope(ax2, simres.traj.time, simres.traj.coords[:,0], simres.traj.coords[:,1],
                          alpha=0.7, lw=1))

  plt.sca(ax3)
  plt.grid(True)
  add_annotation(R'$\psi$', [0.85, 0.78])
  plt.plot(q_ref[:,0], q_ref[:,2], lw=3, color='red', ls='--', zorder=5)
  set_pi_yticks('1/8', fontsize=16)
  animators.append(AnimScope(ax3, simres.traj.time, simres.traj.coords[:,0], simres.traj.coords[:,2],
                          alpha=0.7, lw=1))

  plt.sca(ax4)
  plt.grid(True)
  add_annotation(R'$\dot x$', [0.85, 0.78])
  plt.xticks([-0.2, -0.1, 0., 0.1])
  plt.plot(ref_traj.coords[:,0], ref_traj.vels[:,0], lw=3, color='red', ls='--', zorder=5)
  animators.append(AnimScope(ax4, simres.traj.time, simres.traj.coords[:,0], simres.traj.vels[:,0], 
                          alpha=0.7, lw=1))

  theta = simres.ctrl_state[:,0]
  cont_angle(theta)
  u_ref = trans_coords.usp(theta)
  xi = simres.ctrl_state[:,1:6]

  plt.sca(ax5)
  plt.grid(True)
  plt.plot(theta, u_ref[:,0], ls='--', lw=2, color='#808080')
  animators.append(AnimScope(ax5, simres.traj.time, theta, simres.traj.control[:,0], 
                          alpha=0.7, lw=1))
  add_annotation('thrust', [0.90, 0.75], fontsize=16)

  plt.sca(ax6)
  plt.grid(True)
  plt.plot(theta, u_ref[:,1], ls='--', lw=2, color='#808080')
  animators.append(AnimScope(ax6, simres.traj.time, theta, simres.traj.control[:,1], 
                          alpha=0.7, lw=1))
  add_annotation('torque', [0.90, 0.75], fontsize=16)

  plt.sca(ax7)
  plt.grid(True)
  set_pi_xticks('1', fontsize=16)
  animators.append(AnimScope(ax7, simres.traj.time, theta, xi,  alpha=0.7, lw=1))
  add_annotation('transverse coordinates', [0.70, 0.75], fontsize=16)
  plt.xlabel('trajectory projection', fontsize=16)

  plt.tight_layout(h_pad=1, pad=1)
  anim = Animate(fig, animators, simres.traj.time[-1], fps=60, dpi=120, videopath='fig/pvtol-45deg-oscillations.mp4')
  # anim = Animate(fig, animators, simres.traj.time[-1])

  plt.show()

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  simulate_closed_loop_dynamics()
