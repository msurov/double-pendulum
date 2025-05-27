import numpy as np
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from common.geom_utils import enlarge_rect
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam
)
import casadi as ca
import matplotlib.pyplot as plt
from double_pendulum.anim import draw, animate
from singular_motion_planner.singular_constrs import get_sing_constr_at
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
import scienceplots
from fractions import Fraction
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter
from common.plots import set_pi_xticks, set_pi_yticks



def add_annotation(text : str, textpos : Tuple[int, int]):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.8
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'textcoords': 'axes fraction',
    'font': {
      'size': 18
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)


def show_oscillatory_configurations():
  plt.figure('configuration space', figsize=(4, 4))
  params = dict(color='lightblue', lw=2, alpha=0.8)
  eps = 0.1

  # 0..pi/2
  q2 = np.linspace(eps, np.pi/2 - eps)
  q1_lower = np.pi - q2 + eps
  q1_upper = 2*np.pi - q2 - eps
  plt.fill_between(q2, q1_lower, q1_upper, **params)

  # pi/2..pi
  q2 = np.linspace(np.pi/2 + eps, np.pi - eps)
  q1_lower = 2*np.pi - q2 + eps
  q1_upper = 3*np.pi - q2 - eps
  plt.fill_between(q2, q1_lower, q1_upper, **params)

  q2 = np.linspace(np.pi/2 + eps, np.pi - eps)
  q1_lower = 0*np.pi - q2 + eps
  q1_upper = np.pi - q2 - eps
  plt.fill_between(q2, q1_lower, q1_upper, **params)

  # pi..3pi/2
  q2 = np.linspace(np.pi + eps, 3*np.pi/2 - eps)
  q1_lower = np.pi - q2 + eps
  q1_upper = 2*np.pi - q2 - eps
  plt.fill_between(q2, q1_lower, q1_upper, **params)

  q2 = np.linspace(np.pi + eps, 3*np.pi/2 - eps)
  q1_lower = 3*np.pi - q2 + eps
  q1_upper = 4*np.pi - q2 - eps
  plt.fill_between(q2, q1_lower, q1_upper, **params)

  # 3pi/2..2pi
  q2 = np.linspace(3*np.pi/2 + eps, 2*np.pi - eps)
  q1_lower = 2*np.pi - q2 + eps
  q1_upper = 3*np.pi - q2 - eps
  h2 = plt.fill_between(q2, q1_lower, q1_upper, **params)

  params = dict(color='brown', lw=2, alpha=0.8)
  q1 = np.array([np.pi/2 + eps, 3*np.pi/2 - eps])

  q2 = 2*np.pi - q1
  h1, = plt.plot(q1, q2, **params)

  q1 += np.pi
  q2 = 2*np.pi - q1
  h1, = plt.plot(q1, q2, **params)

  q1 -= 2*np.pi
  q2 = 2*np.pi - q1
  h1, = plt.plot(q1, q2, **params)

  q1 = np.array([-np.pi/2 + eps, np.pi/2 - eps])
  q2 = -q1
  h1, = plt.plot(q1, q2, **params)

  q1 += 2*np.pi
  q2 += 2*np.pi
  h1, = plt.plot(q1, q2, **params)

  q1 = np.linspace(0, 2*np.pi)
  q2 = np.pi - q1
  h1, = plt.plot(q1, q2, **params)
  q2 = 3*np.pi - q1
  h1, = plt.plot(q1, q2, **params)

  ax = plt.gca()
  ax.set_aspect(1)
  plt.xlim(0 - eps, 2*np.pi + eps)
  plt.ylim(0 - eps, 2*np.pi + eps)
  add_annotation(R'$q_1$', [0.35, -0.12])
  add_annotation(R'$q_2$', [-0.15, 0.35])

  set_pi_xticks('1/2')
  set_pi_yticks('1/2')

  plt.grid(True, ls='--')
  plt.tight_layout(pad=0.3)
  plt.savefig('fig/configurations_with_oscillations.pdf')
  plt.show()

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.size": 14,
      "font.family": "Helvetica"
  })

  np.set_printoptions(suppress=True)
  show_oscillatory_configurations()