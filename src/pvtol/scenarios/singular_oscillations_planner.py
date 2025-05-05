import numpy as np
from common.trajectory import Trajectory
from pvtol.dynamics import PVTOLAircraftDynamics
import matplotlib.pyplot as plt
from common.numpy_utils import map_array
from pvtol.anim.draw import (
  draw,
  compute_occupancy_box,
  expand_box
)
from common.plots import set_pi_xticks, set_pi_yticks
from typing import Tuple
from pvtol.scenarios.sample_data import (
  make_sample_data,
  compute_45deg_oscillations,
  compute_circular_traj
)
from pvtol.scenarios.plot_phase import plot_singular_phase_portrait


def motion_schematic(ax, constr, theta_arr, vtol_size):
  configurations = map_array(constr, theta_arr, (3,))
  box = compute_occupancy_box(configurations, vtol_size)
  xmin, xmax, ymin, ymax = expand_box(*box, 10)
  models = [draw(ax, c, vtol_size) for c in configurations]
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  return models

def add_annotation(text : str, textpos : Tuple[int, int]):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.8
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': 20
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def show_trajectory(traj):
  fig, axes = plt.subplots(2, 2, sharex=True)

  plt.sca(axes[0, 0])
  plt.grid(True)
  add_annotation(R'$z$', [-0.18, 0.65])
  plt.plot(traj.coords[:,0], traj.coords[:,1])

  plt.sca(axes[1, 0])
  plt.grid(True)
  set_pi_yticks('1/8')
  add_annotation(R'$x$', [0.68, -0.18])
  add_annotation(R'$\psi$', [-0.18, 0.62])
  plt.plot(traj.coords[:,0], traj.coords[:,2])

  plt.sca(axes[0, 1])
  plt.grid(True)
  add_annotation(R'$\dot x$', [-0.18, 0.62])
  plt.plot(traj.coords[:,0], traj.vels[:,0])

  plt.sca(axes[1, 1])
  plt.grid(True)
  add_annotation(R'$x$', [0.68, -0.18])
  plt.plot(traj.coords[:,0], traj.control)
  plt.legend(['$u_1$', '$u_2$'])

  plt.tight_layout(h_pad=-0.1, pad=0.4)

def show_sample_traj():
  data = make_sample_data('tictoc')
  traj = data['traj']
  constr = data['constr']

  _,ax = plt.subplots(1, 1, figsize=(6, 4))
  ax.set_aspect(1)
  plt.plot(traj.coords[:,0], traj.coords[:,1], zorder=-1)

  show_trajectory(traj)

  plt.figure('phase')
  reduced = data['reduced']
  reduced_traj = data['reduced_traj']
  plot_singular_phase_portrait(reduced, reduced_traj, 0.)
  plt.tight_layout()

  plt.show()

def show_tictoc_schematically():
  dynamics = PVTOLAircraftDynamics()
  data = compute_circular_traj(dynamics)
  traj = data['reduced_traj']
  constr = data['constr']

  theta_min = np.min(data['reduced_traj'].coords)
  theta_max = np.max(data['reduced_traj'].coords)
  theta = np.linspace(theta_min, theta_max)
  q = map_array(constr, theta)

  _,ax = plt.subplots(1, 1, figsize=(7, 3.3))
  ax.set_aspect(1)
  plt.plot(q[:,0], q[:,1], '--', lw=2, zorder=-1)
  motion_schematic(ax, constr, [theta_min, 0., theta_max], 1.5)
  plt.axis('off')
  plt.xticks([])
  plt.yticks([])

  plt.tight_layout(pad=-1)
  plt.savefig('fig/pvtol_tic_toc_schematically.svg')
  plt.show()

def show_45deg_oscillation_schematically():
  dynamics = PVTOLAircraftDynamics()
  data = compute_45deg_oscillations(dynamics)
  constr = data['constr']
  traj = data['reduced_traj']
  theta_min = np.min(traj.coords)
  theta_max = np.max(traj.coords)
  theta = np.linspace(theta_min, theta_max)
  q = map_array(constr, theta)

  _,ax = plt.subplots(1, 1, figsize=(4.5, 4))
  ax.set_aspect(1)
  plt.plot(q[:,0], q[:,1], '--', lw=2, zorder=-1)
  motion_schematic(ax, constr, [theta_min, 0., theta_max], 0.5)
  plt.axis('off')
  plt.xticks([])
  plt.yticks([])

  plt.tight_layout(pad=-1)
  plt.savefig('fig/pvtol_45deg_oscillations_schematically.svg')
  plt.show()

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.size": 16,
      "font.family": "Helvetica"
  })

  np.set_printoptions(suppress=True)
  show_sample_traj()
  show_tictoc_schematically()
  show_45deg_oscillation_schematically()
