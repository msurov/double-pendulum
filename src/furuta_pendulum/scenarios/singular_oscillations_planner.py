from furuta_pendulum.dynamics import (
  FurutaPendulumDynamics,
  FurutaPendulumPar,
  furuta_pendulum_param_default
)
from furuta_pendulum.anim import draw, animate, motion_schematic
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from singular_motion_planner.singular_constrs import get_sing_constr_at
from singular_motion_planner.plots import show_reduced_dynamics_phase_prortrait
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


def show_trajectory_projections(traj : Trajectory, savetofile=None):
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

def show_trajectory(traj : Trajectory, savetofile=None):
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

  show_trajectory(tr_orig)
  show_reduced_dynamics_phase_prortrait(reduced, tr_closed)
  motion_schematic(tr_orig, par)

def main():
  pass

def show_sample_traj():
  par = furuta_pendulum_param_default
  dynamics = FurutaPendulumDynamics(par)
  singpt = [4, -2.]

  constr = get_sing_constr_at(dynamics, singpt, scale=1000)
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-0.4, -1e-3], 0.0, max_step=1e-3)
  tr_right = solve_reduced(reduced, [0.4, 1e-3], 0.0, max_step=1e-3)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_reduced = traj_repeat(tr_closed, 2)
  print(tr_reduced.time[-1])
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)

  show_trajectory(tr_orig)
  show_trajectory_projections(tr_orig)
  show_reduced_dynamics_phase_prortrait(reduced, tr_closed)
  motion_schematic(tr_orig, par)
  # a = animate(tr_orig, par, speedup=0.2)

  plt.show()

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.size": 14,
      "font.family": "Helvetica"
  })

  np.set_printoptions(suppress=True)
  # main()
  show_sample_traj()
