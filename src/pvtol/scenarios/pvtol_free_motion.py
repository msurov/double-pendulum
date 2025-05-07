from pvtol.dynamics import PVTOLAircraftDynamics
from visualization.anim import Animate
from pvtol.anim import AnimPVTOLAircraft, AnimPVTOLAircraftPar
from pvtol.anim.draw import compute_occupancy_box, expand_box
from common.mechsys_integ import integrate
import numpy as np
import matplotlib.pyplot as plt


def free_motion_test():
  dynamics = PVTOLAircraftDynamics()
  def ctrl(t, q, dq):
    return [1.2, 0.2]

  q0 = np.zeros(3)
  dq0 = np.zeros(3)
  traj = integrate(dynamics, q0, dq0, 6., ctrl, max_step=1e-2)

  fig = plt.figure('PVTOL anim', figsize=(9, 6))
  ax = plt.gca()
  ax.set_aspect(1)

  anim_par = AnimPVTOLAircraftPar(aircraft_size=2)
  anim_pvtol = AnimPVTOLAircraft(ax, traj, anim_par)
  box = compute_occupancy_box(traj.coords, anim_par.aircraft_size)
  xmin, xmax, ymin, ymax = expand_box(*box, 10)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)

  plt.tight_layout(h_pad=0, pad=0)
  # anim = Animate(fig, animators, traj.time[-1], fps=60, dpi=120, videopath='fig/pvtol-free-motion.mp4')
  anim = Animate(fig, [anim_pvtol], traj.time[-1], fps=30)

  plt.show()

if __name__ == '__main__':
  free_motion_test()
