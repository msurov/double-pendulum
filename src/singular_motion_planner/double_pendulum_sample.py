import numpy as np
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default
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
from typing import List, Tuple


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

def enlarge_rect(r, coef):
  c = np.mean(r, axis=1)
  w = r[:,1] - r[:,0]
  return np.array([c - w * coef / 2, c + w * coef / 2]).T

def get_traj_bounding_rect(traj : Trajectory):
  qmin = np.min(traj.coords, axis=0)
  qmax = np.max(traj.coords, axis=0)
  return np.array([qmin, qmax]).T

def get_cartesian_rect(traj : Trajectory, par : DoublePendulumParam):
  q1, q2 = traj.coords.T
  x1 = par.lengths[0] * np.sin(q1)
  y1 = par.lengths[0] * np.cos(q1)
  x2 = x1 + par.lengths[0] * np.sin(q1 + q2)
  y2 = y1 + par.lengths[0] * np.cos(q1 + q2)

  xmin = min(0, np.min(x1))
  xmin = min(xmin, np.min(x2))

  xmax = max(0, np.max(x1))
  xmax = max(xmax, np.max(x2))

  ymin = min(0, np.min(y1))
  ymin = min(ymin, np.min(y2))

  ymax = max(0, np.max(y1))
  ymax = max(ymax, np.max(y2))

  return np.array([
    [xmin, xmax],
    [ymin, ymax]
  ])

def motion_schematic(traj : Trajectory, par : DoublePendulumParam, savetofile=None):
  d = traj.phase - traj.phase[0]
  d = np.linalg.norm(d, axis=1)
  i, = np.nonzero(d < 1e-5)
  i = i[1]
  q1 = traj.coords[0]
  q2 = traj.coords[i//4]
  q3 = traj.coords[i//2]

  fig,ax = plt.subplots(1, 1, num=f'schematic at {q1[0]:.2f}, {q1[1]:.2f}', figsize=(6, 4))
  ax.set_aspect(1)
  draw(q1, par, alpha=1, color='#3030E0', linewidth=2)
  draw(q2, par, alpha=1, color='#3030C0', linewidth=2)
  draw(q3, par, alpha=1, color='#3030A0', linewidth=2)
  r = get_cartesian_rect(traj, par)
  xdiap, ydiap = enlarge_rect(r, 1.05)
  ax.set_xlim(*xdiap)
  ax.set_ylim(*ydiap)

  plt.grid(True)
  plt.tight_layout()
  if savetofile is not None:
    plt.savefig(savetofile)

def show_phase_prortrait(reduced : ReducedDynamics, reduced_traj : Trajectory, savetofile=None):
  sleft = np.min(reduced_traj.coords)
  sright = np.max(reduced_traj.coords)
  dsmin = np.min(reduced_traj.vels)
  dsmax = np.max(reduced_traj.vels)

  # with plt.style.context('science'):
  #   plt.figure('phase', figsize=(6, 4))
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

def show_trajectory_projections(traj : Trajectory, reduced_traj : Trajectory, savetofile=None):
  q0 = traj.coords[0]
  fig, axes = plt.subplots(2, 2, sharex=True, num=f'trajectory projections at {q0[0]:.2f}, {q0[1]:.2f}')
  ax = axes[0,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.coords[:,0], traj.coords[:,1])
  plt.ylabel(R'$q_2$')

  ax = axes[0,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.coords[:,0], traj.vels[:,0])
  plt.ylabel(R'$\dot q_1$')

  ax = axes[1,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.coords[:,0], traj.vels[:,1])
  plt.xlabel(R'$q_1$')
  plt.ylabel(R'$\dot q_2$')

  ax = axes[1,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.coords[:,0], traj.control)
  plt.xlabel(R'$q_1$')
  plt.ylabel(R'$u$')

  plt.tight_layout()

  if savetofile is not None:
    plt.savefig(savetofile)

def show_trajectory(traj : Trajectory, reduced_traj : Trajectory, savetofile=None):
  q0 = traj.coords[0]
  fig, axes = plt.subplots(2, 2, sharex=True, num=f'trajectory at {q0[0]:.2f}, {q0[1]:.2f}')
  ax = axes[0,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.time, traj.coords[:,1])
  plt.ylabel(R'$q_2$')

  ax = axes[0,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.time, traj.vels[:,0])
  plt.ylabel(R'$\dot q_1$')

  ax = axes[1,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.time, traj.vels[:,1])
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\dot q_2$')

  ax = axes[1,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.time, traj.control)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$u$')

  plt.tight_layout()

  if savetofile is not None:
    plt.savefig(savetofile)

def process_sing_traj_at_sing_point(singpt):
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  constr = get_sing_constr_at(dynamics, singpt)
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-0.02, -1e-4], 0.0, max_step=1e-4)
  tr_right = solve_reduced(reduced, [0.02, 1e-4], 0.0, max_step=1e-4)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_reduced = traj_repeat(tr_closed, 2)
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)

  show_trajectory(tr_orig, tr_reduced)
  show_phase_prortrait(reduced, tr_closed)
  motion_schematic(tr_orig, par)

def main():
  positions = [
    [-2.045583727546234, 0.6462469356355233],
    [-1.7050253662106756, -3.30435292667277],
    [11.690270453301533, 2.0109207709898262],
    [1.0, -2.5],
    [2.5, 1.3],
    [-1, 2.5708],
    [-1, 2.4],
    [-1, 2.7],
    [-2, 2.8],
    [-0.5, 2.5],
    [-1, 0.6],
    [-1.2, 0.8],
    [-2, 0.9],
    [-2.5, 2.8],
    [ 2.35619449, -0.78539816],
    [-1.04719755,  2.61799388]
  ]
  for pos in positions:
    process_sing_traj_at_sing_point(pos)
    # plt.pause(0.001)
    plt.show()

  plt.show()

def show_sample_traj():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  singpt = np.array([-2.2, 1.12])
  constr = get_sing_constr_at(dynamics, singpt)
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-0.05, -1e-4], 0.0, max_step=1e-4)
  tr_right = solve_reduced(reduced, [0.08, 1e-4], 0.0, max_step=1e-4)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_reduced = traj_repeat(tr_closed, 2)
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)

  show_trajectory(tr_orig, tr_reduced)
  show_phase_prortrait(reduced, tr_closed)
  motion_schematic(tr_orig, par)
  plt.show()

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.size": 14,
      "font.family": "Helvetica"
  })

  np.set_printoptions(suppress=True)
  main()
  # show_sample_traj()
