import numpy as np
import matplotlib.pyplot as plt
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
)
from common.geom_utils import enlarge_rect
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default
)
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
)
from singular_motion_planner.singular_constrs import get_sing_constr_at


def plot_integral_curve(rudced, s1, s2, ds1, 
                        alpha=0.8, 
                        max_integ_step=1e-2,
                        color='steelblue', ls='-', lw=0.5, **kwargs):
  
  traj = solve_reduced(rudced, [s1, s2], ds1, max_step=max_integ_step)

  params = {
    'ls': ls, 'color': color, 'lw': lw, 'alpha': alpha
  }
  s = traj.coords
  ds = traj.vels
  lines = plt.plot(s, ds, **params, **kwargs)
  lines += plt.plot(s, -ds, **params, **kwargs)
  return lines

def plot_singular_phase_portrait(
    reduced : ReducedDynamics,
    reference_traj : Trajectory,
    s_singular : float,
    wider = 0.2,
    higher = 0.2,
    npts_x = 25,
    npts_y = 10,
  ):

  ax = plt.gca()
  plt.axhline(0, ls='--', color='black', lw=0.5)
  plt.axvline(0, ls='--', color='black', lw=0.5)

  assert wider >= 0
  assert higher >= 0
  assert npts_x > 2
  assert npts_y > 2

  inc_height = 1 + higher
  inc_width = 1 + wider

  smin = np.min(reference_traj.coords)
  smax = np.max(reference_traj.coords)
  dsmin = np.min(reference_traj.vels)
  dsmax = np.max(reference_traj.vels)
  eps = 1e-3
  integ_step = 1e-3 * (smax - smin)

  step = (smax - smin) * inc_width / (npts_x - 2)
  assert step > 0
  nleft_inside = int((s_singular - smin) / step + 0.5)
  nleft_outside = int((inc_width - 1) * (smax - smin) / step + 0.5)
  nright_inside = int((smax - s_singular) / step + 0.5)
  nright_outside = int((inc_width - 1) * (smax - smin) / step + 0.5)

  ny = int(npts_y / 2 + 0.5)

  plot_args = {
    'ls': '-',
    'lw': 1,
    'alpha': 0.5,
    'color': '#6060F0',
    'max_integ_step': integ_step
  }

  # left to sing
  for left in np.linspace(smin, s_singular - eps, nleft_inside)[1:-1]:
    plot_integral_curve(reduced, left, s_singular - eps, eps, **plot_args)
    plot_integral_curve(reduced, left, s_singular - eps, inc_height * dsmax, **plot_args)

  # left to sing, close to sing
  left = s_singular - 0.01 * (smax - smin)
  plot_integral_curve(reduced, left, s_singular - eps, eps, **plot_args)
  plot_integral_curve(reduced, left, s_singular - eps, inc_height * dsmax, **plot_args)

  # left to sing, outside the traj
  for left in np.linspace(smax - inc_width * (smax - smin), smin, nleft_outside)[1:-1]:
    plot_integral_curve(reduced, left, s_singular - eps, eps, **plot_args)
    plot_integral_curve(reduced, left, s_singular - eps, inc_height * dsmax, **plot_args)

  # left to sing, starting from left border
  left = smax - inc_width * (smax - smin)
  for ds in np.sqrt(np.linspace(eps**2, (inc_height * dsmax)**2, ny)):
    plot_integral_curve(reduced, left, s_singular - eps, ds, **plot_args)

  # right to singularity
  for right in np.linspace(smax, s_singular + eps, nright_inside)[1:-1]:
    plot_integral_curve(reduced, right, s_singular + eps, eps, **plot_args)
    plot_integral_curve(reduced, right, s_singular + eps, inc_height * dsmax, **plot_args)

  # right to singularity, close to singularity
  right = s_singular + 0.01 * (smax - smin)
  plot_integral_curve(reduced, right, s_singular + eps, eps, **plot_args)
  plot_integral_curve(reduced, right, s_singular + eps, inc_height * dsmax, **plot_args)

  # right to singularity, outside the traj
  for right in np.linspace(smin + inc_width * (smax - smin), smax, nright_outside)[1:-1]:
    plot_integral_curve(reduced, right, s_singular + eps, eps, **plot_args)
    plot_integral_curve(reduced, right, s_singular + eps, inc_height * dsmax, **plot_args)

  # right to sing, starting from right border
  right = smin + inc_width * (smax - smin)
  for ds in np.sqrt(np.linspace(eps**2, (inc_height * dsmax)**2, ny)):
    plot_integral_curve(reduced, right, s_singular + eps, ds, **plot_args)

  plot_ref_args = {
    'ls': '-',
    'lw': 3,
    'alpha': 0.8,
    'color': '#B06060'
  }
  plt.plot(reference_traj.coords, reference_traj.vels, **plot_ref_args)

  bbox = {
        'boxstyle': 'round',
        'fc': 'white',
        'ec': 'white',
      }
  ax.annotate(
    R'$\dot\phi$', xy=(-0.05, 0.45), xytext=(-15,2), 
    ha='left', va='top', 
    xycoords='axes fraction',
    textcoords='offset points',
    fontsize=18,
    bbox = bbox,
  )
  ax.annotate(
    R'$\phi$', xy = (0.48, -0.06), xytext = (-15,2), 
    ha = 'left', va = 'top', 
    xycoords = 'axes fraction',
    textcoords = 'offset points',
    fontsize = 18,
    bbox = bbox,
  )

  ax.set_xlim(smax - inc_width * (smax - smin), smin + inc_width * (smax - smin))
  ax.set_ylim(-dsmax * inc_height, dsmax * inc_height)
  plt.tight_layout()
  return plt.gcf()
