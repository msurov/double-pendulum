import numpy as np
from common.trajectory import Trajectory
from pvtol.dynamics import PVTOLAircraftDynamics
import matplotlib.pyplot as plt
from common.numpy_utils import map_array
from pvtol.anim.draw import draw
from singular_motion_planner.plots import (
  show_reduced_dynamics_phase_prortrait
)
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
from common.plots import set_pi_xticks, set_pi_yticks
from typing import Tuple
from pvtol.scenarios.sample_data import make_sample_data
from pvtol.scenarios.plot_phase import plot_singular_phase_portrait


def expand_box(xmin, xmax, ymin, ymax, pcnt):
  xc = (xmax + xmin) / 2
  w = (xmax - xmin) * (100 + pcnt) / 100
  yc = (ymax + ymin) / 2
  h = (ymax - ymin) * (100 + pcnt) / 100
  return (
    xc - w / 2,
    xc + w / 2,
    yc - h / 2,
    yc + h / 2,
  )

def compute_occupancy_box(q_arr, vtol_size):
  pts = []
  for q in q_arr:
    p1 = np.array([
      q[0] + np.cos(q[2]) * 0.5 * vtol_size,
      q[1] + np.sin(q[2]) * 0.5 * vtol_size,
    ])
    p2 = np.array([
      q[0] - np.cos(q[2]) * 0.5 * vtol_size,
      q[1] - np.sin(q[2]) * 0.5 * vtol_size,
    ])
    pts.append(p1)
    pts.append(p2)
  pts = np.array(pts)
  print(pts)
  xmin,ymin = np.min(pts, axis=0)
  xmax,ymax = np.max(pts, axis=0)
  return xmin, xmax, ymin, ymax

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
  data = make_sample_data()
  traj = data['traj']
  constr = data['constr']

  _,ax = plt.subplots(1, 1, figsize=(6, 4))
  ax.set_aspect(1)
  plt.plot(traj.coords[:,0], traj.coords[:,1], zorder=-1)
  theta_min = np.min(data['reduced_traj'].coords)
  theta_max = np.max(data['reduced_traj'].coords)
  motion_schematic(ax, constr, [theta_min, 0., theta_max], 2.0)

  show_trajectory(traj)

  plt.figure('phase')
  reduced = data['reduced']
  reduced_traj = data['reduced_traj']
  plot_singular_phase_portrait(reduced, reduced_traj, 0.)
  plt.tight_layout()

  plt.show()

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.size": 16,
      "font.family": "Helvetica"
  })

  np.set_printoptions(suppress=True)
  show_sample_traj()
