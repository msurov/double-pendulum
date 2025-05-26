from double_pendulum.anim import DoublePendulumAnim, get_view_parameters
from double_pendulum.dynamics import double_pendulum_param_default
from visualization import (
  AnimGraph,
  AnimScope,
  Animate
)
import matplotlib.pyplot as plt
import numpy as np
from common.trajectory import Trajectory

def test():
  fig,ax = plt.subplots(1, 1)
  view_par = get_view_parameters(double_pendulum_param_default)
  t = np.linspace(0, 5, 100)
  phase = np.array([
    np.sin(t),
    np.cos(t),
    0 * t,
    0 * t
  ]).T
  traj_sim = Trajectory(
    time = t,
    phase = phase
  )

  animators = [
    DoublePendulumAnim(ax, view_par, traj_sim, shadow_color='#F0C0C0')
  ]

  fig.tight_layout()

  anim = Animate(fig, animators, traj_sim.time[-1], 30, 1)
  plt.show()

if __name__ == '__main__':
  test()
