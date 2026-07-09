from ball_and_beam.dynamics import BallAndBeamDynamics, ball_and_beam_parameters_default
from ball_and_beam.anim import launch_anim, get_vis_par
import casadi as ca
from common.numpy_utils import map_array, find_all_roots
import numpy as np
import matplotlib.pyplot as plt
from common.plots import set_pi_xticks
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from common.mechsys_integ import integrate


def main():
  par = ball_and_beam_parameters_default
  dynamics = BallAndBeamDynamics(par, auto_compute=False)
  vispar = get_vis_par(par)

  traj = integrate(dynamics, [0., 0.05], [0., 0.0], 2., max_step=0.01)
  a = launch_anim(traj, vispar, speedup=0.5)
  plt.show()

if __name__ == '__main__':
  main()
