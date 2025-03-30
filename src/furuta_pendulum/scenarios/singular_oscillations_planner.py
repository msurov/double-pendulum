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
from common.plots import (
  set_pi_xticks,
  set_pi_yticks
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
  plt.plot(traj.coords[:,0], traj.coords[:,1], alpha=0.8)
  plt.ylabel(R'$q_2$')
  set_pi_yticks('1/12')

  ax = axes[0,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.coords[:,0], traj.vels[:,0], alpha=0.8)
  plt.ylabel(R'$\dot q_1$')

  ax = axes[1,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.coords[:,0], traj.vels[:,1], alpha=0.8)
  plt.xlabel(R'$q_1$')
  set_pi_xticks('1/6')
  plt.ylabel(R'$\dot q_2$')

  ax = axes[1,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.coords[:,0], traj.control, alpha=0.8)
  plt.xlabel(R'$q_1$')
  set_pi_xticks('1/6')
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
  plt.plot(traj.time, traj.coords[:,1], alpha=0.8)
  plt.ylabel(R'$q_2$')
  set_pi_yticks('1/12')

  ax = axes[0,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.time, traj.vels[:,0], alpha=0.8)
  plt.ylabel(R'$\dot q_1$')

  ax = axes[1,0]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.time, traj.vels[:,1], alpha=0.8)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\dot q_2$')

  ax = axes[1,1]
  plt.sca(ax)
  plt.grid(True)
  plt.plot(traj.time, traj.control, alpha=0.8)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$u$')

  plt.tight_layout()

  if savetofile is not None:
    plt.savefig(savetofile)

def show_sample_traj():
  par = FurutaPendulumPar(
    link_1_mass = 0.100,
    link_2_mass = 0.025,
    link_1_length = 0.1,
    link_2_length = 0.2,
    link_1_inertia_tensor = np.diag([0., 0.00015, 0.00015]),
    link_2_inertia_tensor = np.diag([0., 0.00015, 0.00015]),
    link_1_mass_center = np.array([0.01, 0., 0.]),
    link_2_mass_center = np.array([0.12, 0., 0.]),
    link_1_orient = np.eye(3),
    link_2_orient = np.array([
      [0.,  0., 1.],
      [0., -1., 0.],
      [1.,  0., 0.]
    ]),
    joint_1_pos = np.array([0., 0., 0.]),
    joint_2_pos = np.array([0.1, 0., 0.]),
    gravity_accel = 9.81
  )
  dynamics = FurutaPendulumDynamics(par)
  singpt = [-2., -2.2]

  constr = get_sing_constr_at(dynamics, singpt, scale=1000)
  assert constr is not None
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-0.8, -0.5e-3], 0.0, max_step=1e-3)
  tr_right = solve_reduced(reduced, [0.8, 0.5e-3], 0.0, max_step=1e-3)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_reduced = traj_repeat(tr_closed, 2)
  print('period = ', tr_closed.time[-1])
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)

  show_trajectory(tr_orig)
  show_trajectory_projections(tr_orig)
  show_reduced_dynamics_phase_prortrait(reduced, tr_closed)
  # motion_schematic(tr_orig, par)
  a = animate(tr_orig, par, speedup=0.2)

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
