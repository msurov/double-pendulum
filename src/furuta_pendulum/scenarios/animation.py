from furuta_pendulum.anim.animate import animate
from furuta_pendulum.dynamics.parameters import furuta_pendulum_param_default
from common.trajectory import Trajectory
import matplotlib.pyplot as plt
import numpy as np


def main():
  par = furuta_pendulum_param_default
  t = np.linspace(0, 5, 100)
  q = 2 * np.pi * np.array([np.sin(t), np.cos(t)]).T
  dq = 2 * np.pi * np.array([np.cos(t), -np.sin(t)]).T
  phase = np.concatenate((q, dq), axis=1)
  traj = Trajectory(
    time = t,
    phase = phase
  )
  anim = animate(traj, par)
  plt.show()

if __name__ == '__main__':
  main()
