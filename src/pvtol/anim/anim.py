from .draw import PVTOLView
from common.trajectory import Trajectory
from dataclasses import dataclass
from scipy.interpolate import make_interp_spline

@dataclass
class AnimPVTOLAircraftPar:
  aircraft_size : float

class AnimPVTOLAircraft:
  def __init__(self, ax, traj : Trajectory, anim_par : AnimPVTOLAircraftPar):
    self.pvtol_view = PVTOLView(ax, size=anim_par.aircraft_size)
    self.traj_sp = make_interp_spline(
      traj.time, traj.coords, k=1
    )
    self.tspan = traj.time[0], traj.time[-1]
  
  def update(self, t):
    t1, t2 = self.tspan
    t = max(t, t1)
    t = t1 + (t - t1) % (t2 - t1)
    q = self.traj_sp(t)
    self.pvtol_view.move(q)

  @property
  def patches(self):
    return self.pvtol_view.patches
