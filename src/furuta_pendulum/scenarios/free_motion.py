from furuta_pendulum.dynamics import (
  FurutaPendulumDynamics,
  FurutaPendulumPar,
  furuta_pendulum_param_default
)
from common.mechsys_integ import integrate
from furuta_pendulum.anim import draw, animate, motion_schematic
import matplotlib.pyplot as plt
import numpy as np


def main():
  par = furuta_pendulum_param_default
  par.joint_1_friction = 0.002
  par.joint_2_friction = 0.0005
  dynamics = FurutaPendulumDynamics(par)
  q0 = np.array([0., 0.01])
  dq0 = np.array([0., 0])
  traj = integrate(dynamics, q0, dq0, 7, max_step=1e-2)

  a = animate(traj, par, speedup=1.)
  plt.show()

if __name__ == '__main__':
  main()
