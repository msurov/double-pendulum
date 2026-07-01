import matplotlib.pyplot as plt
from common.trajectory import Trajectory, make_traj, traj_repeat
from ball_and_beam.dynamics import BallAndBeamDynamics, ball_and_beam_parameters_default
import casadi as ca
from common.numpy_utils import map_array, find_all_roots
from common.mechsys import MechanicalSystem
from common.trajectory import Trajectory, make_traj
import numpy as np
from common.plots import set_pi_xticks
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from singular_motion_planner.reduced_dynamics import (
  reconstruct_trajectory,
  ReducedDynamics
)
from typing import Optional, Tuple
from ball_and_beam.anim import launch_anim, get_vis_par, BallAndBeamVisPar
from ball_and_beam.scenarios.motion_planning import compute_periodic_trajectory


def main():
  par = ball_and_beam_parameters_default
  dynamics = BallAndBeamDynamics(par, auto_compute=False)
  traj = compute_periodic_trajectory(dynamics)

  _, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 4), num='traj')
  plt.sca(axes[0,0])
  plt.plot(traj.coords[:,1], traj.vels[:,0])
  plt.xlabel('s')
  plt.ylabel('theta')
  plt.grid(True)

  plt.sca(axes[0,1])
  plt.plot(traj.coords[:,1], traj.vels[:,1])
  plt.xlabel('s')
  plt.ylabel('ds')
  plt.grid(True)

  plt.sca(axes[1,0])
  plt.plot(traj.coords[:,1], traj.coords[:,0])
  plt.xlabel('s')
  plt.ylabel('theta')
  plt.grid(True)

  plt.sca(axes[1,1])
  plt.plot(traj.coords[:,1], traj.control)
  plt.xlabel('s')
  plt.ylabel('u')
  plt.grid(True)

  plt.tight_layout(pad=0.1)
  plt.pause(0.01)
  
  vispar = BallAndBeamVisPar(
    ball_radius = par.ball_radius,
    beam_thickness = 0.02,
    beam_length = 0.9,
    joint_radius = 0.01,
    surface_vertical_displacement = par.ball_center_displacement - par.ball_radius
  )
  get_vis_par(par)
  traj4 = traj_repeat(traj, 4)
  plt.figure(figsize=(10, 4), num='ball and beam anim')
  a = launch_anim(traj4, vispar, speedup=0.25, fps=60)
  plt.tight_layout(pad=0.5)
  # a.save('dump/ball_and_beam.mp4', dpi=150)

  plt.show()

if __name__ == '__main__':
  main()
