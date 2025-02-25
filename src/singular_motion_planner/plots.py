import numpy as np
from common.trajectory import Trajectory
import matplotlib.pyplot as plt
from singular_motion_planner.reduced_dynamics import ReducedDynamics
import scienceplots
from typing import Tuple


def add_annotation(text : str, textpos : Tuple[int, int]):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.8
  }
  annotate_par = {
    'xycoords': 'axes points',
    'font': {
      'size': 18
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def show_reduced_dynamics_phase_prortrait(reduced : ReducedDynamics, reduced_traj : Trajectory, savetofile=None):
  sleft = np.min(reduced_traj.coords)
  sright = np.max(reduced_traj.coords)
  dsmin = np.min(reduced_traj.vels)
  dsmax = np.max(reduced_traj.vels)

  s0 = reduced_traj.coords[0,0]
  plt.figure(num=f'phase portrait at {s0:.2f}', figsize=(6, 4))
  plt.axhline(0, color='black', alpha=0.5, lw=1)
  plt.axvline(0, color='black', alpha=0.5, lw=1)

  s1 = sleft * 1.2
  s2 = sright * 1.2
  ds1 = dsmin * 1.2
  ds2 = dsmax * 1.2
  s = np.linspace(s1, s2, 30)
  ds = np.linspace(ds1, ds2, 30)
  X,Y = np.meshgrid(s, ds)
  U = np.zeros(X.shape)
  V = np.zeros(X.shape)
  for i in range(len(s)):
    for j in range(len(ds)):
      U[j,i] = ds[j]
      V[j,i] = (-reduced.beta(s[i]) * ds[j]**2 - reduced.gamma(s[i])) / reduced.alpha(s[i])

  plt.streamplot(X, Y, U, V, color='lightblue')
  plt.plot(reduced_traj.coords, reduced_traj.vels, lw=2, color='darkblue', alpha=1)
  plt.gca().set_xlim(s1, s2)
  plt.gca().set_ylim(ds1, ds2)
  add_annotation(R'$\theta$', (340, 10))
  add_annotation(R'$\dot\theta$', (8, 210))
  plt.tight_layout()

  if savetofile is not None:
    plt.savefig(savetofile)
