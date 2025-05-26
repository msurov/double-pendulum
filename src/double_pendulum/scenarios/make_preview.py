from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default,
  convert_parameters,
  double_pendulum_param_disturbed
)
from double_pendulum.anim import (
  DoublePendulumViewParam,
  DoublePendulumView,
  get_view_parameters
)
from typing import Tuple
import casadi as ca
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  reconstruct_trajectory
)
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat,
  traj_reverse
)
from functools import reduce


def add_annotation(text : str, textpos : Tuple[float, float], fontsize=16):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.8
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': fontsize
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def show_phase_curve(reduced_dynamics : ReducedDynamics, x_diap, dx0):
  def rhs(x, st):
    dx, = st
    alpha = reduced_dynamics.alpha(x)
    beta = reduced_dynamics.beta(x)
    gamma = reduced_dynamics.gamma(x)
    return (-beta * dx**2 - gamma) / (alpha * dx)

  def stop_cond(x, st):
    dx, = st
    if dx <= 0:
      return -1
    return 1
  stop_cond.terminal = True

  sol = solve_ivp(rhs, x_diap, [dx0], max_step=1e-2, events=stop_cond)
  plt.plot(sol.t, sol.y[0], color='#8080C0', alpha=0.8, lw=1)
  plt.plot(sol.t, -sol.y[0], color='#8080C0', alpha=0.8, lw=1)

def get_constr(par : DoublePendulumParam):
  par2 = convert_parameters(par)
  p1,p2,p3,p4,p5 = par2.p
  g = par2.gravity_accel
  qs = ca.DM([3*ca.pi/4, -ca.pi/4])
  k = -p2*p3**2*ca.sin(qs[1]) - 1/3*p2**2*p3*ca.sin(2*qs[1])

  theta = ca.SX.sym('theta')
  q = qs + ca.vertcat(-p3, p3 + p2 * ca.cos(qs[1])) * theta + \
    k / (2 * p3) * ca.vertcat(0, 1) * theta**2

  q_fun = ca.Function('constr', [theta], [q])
  return {
    'constr': q_fun,
    'singular_point': np.reshape(qs, (2,))
  }

def make_sample_traj(dynamics, par):
  constr_data = get_constr(par)
  constr = constr_data['constr']
  reduced = ReducedDynamics(dynamics, constr)
  # verify_reduced_dynamics(par, reduced)
  tr_1 = solve_reduced(reduced, [-4., -1e-3], 0.0, max_step=1e-3)
  tr_2 = solve_reduced(reduced, [1.0, 1e-3], 0.0, max_step=1e-3)[::-1]
  tr_reduced = reduce(traj_join, 
    [
      tr_1,
      tr_2,
      traj_reverse(tr_2),
      traj_reverse(tr_1),
    ]
  )

  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)
  theta_min = np.min(tr_reduced.coords)
  theta_max = np.max(tr_reduced.coords)

  return {
    **constr_data,
    'q_left': np.reshape(constr(theta_min), (2,)),
    'q_right': np.reshape(constr(theta_max), (2,)),
    'reduced_traj': tr_reduced,
    'traj': tr_orig,
    'reduced_dynamics': reduced
  }

def get_vertix_coords(q : np.ndarray, par : DoublePendulumParam):
  l1, l2 = par.lengths
  theta1, theta2 = q
  x1 = l1 * np.sin(theta1)
  y1 = l1 * np.cos(theta1)
  x2 = x1 + l2 * np.sin(theta1 + theta2)
  y2 = y1 + l2 * np.cos(theta1 + theta2)
  xarr = np.array([0, x1, x2])
  yarr = np.array([0, y1, y2])
  return xarr, yarr

def make_preview():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  data = make_sample_traj(dynamics, par)
  reduced_dynamics = data['reduced_dynamics']
  reduced_traj = data['reduced_traj']

  eps = 1e-3
  fig, axes = plt.subplots(1, 2, figsize=(8, 4), width_ratios=(1.2, 1))

  theta_2 = brentq(reduced_dynamics.alpha, 1.8, 2.0)
  theta_2 = float(theta_2)
  dtheta_2 = np.sqrt(-reduced_dynamics.gamma(theta_2) / reduced_dynamics.beta(theta_2))
  dtheta_2 = float(dtheta_2)

  dtheta_1 = np.sqrt(-reduced_dynamics.gamma(0) / reduced_dynamics.beta(0))
  dtheta_1 = float(dtheta_1)

  ax = axes[1]
  plt.sca(ax)
  plt.grid(True, lw=0.5, ls='--')
  plt.axvline(0, ls='--', lw=1, color='#404040')
  plt.axhline(0, ls='--', lw=1, color='#404040')
  plt.axvline(theta_2, ls='--', lw=1, color='#404040')

  if False:
    for x0 in np.linspace(-0.2, -4, 12, endpoint=False):
      show_phase_curve(reduced_dynamics, [x0, -eps], eps)

    for x0 in np.linspace(-0.2, -1.5, 10, endpoint=False):
      show_phase_curve(reduced_dynamics, [x0, -eps], 100)

    for x0 in np.linspace(-5, -4, 3, endpoint=False):
      show_phase_curve(reduced_dynamics, [x0, -eps], eps)

    for dx0 in np.linspace(30, 120, 8, endpoint=False):
      show_phase_curve(reduced_dynamics, [-5, -eps], dx0)

    for x0 in np.linspace(0.2, 1., 3, endpoint=False):
      show_phase_curve(reduced_dynamics, [x0, eps], eps)
      show_phase_curve(reduced_dynamics, [x0, eps], 100)

    for x0 in np.linspace(theta_2 - 0.1, 1., 4):
      show_phase_curve(reduced_dynamics, [x0, eps], eps)
      show_phase_curve(reduced_dynamics, [x0, eps], 100)

    for x0 in np.linspace(theta_2 + 0.1, 3.0, 8):
      show_phase_curve(reduced_dynamics, [x0, 3.0], 100)
      show_phase_curve(reduced_dynamics, [x0, 3.0], eps)

  show_phase_curve(reduced_dynamics, [theta_2 + eps, 3.0], dtheta_2)
  show_phase_curve(reduced_dynamics, [theta_2 - eps, eps], dtheta_2)

  plt.plot(reduced_traj.coords, reduced_traj.vels, lw=2, color='#C04040')
  plt.xlim(-4.5, 2.8)
  plt.ylim(-100, 100)
  plt.yticks([-60, -30, 0, 30, 60], [])
  plt.xticks(np.arange(-4, 3), [])
  # add_annotation(R'$\theta$', [0.5, -0.15], fontsize=18)
  # add_annotation(R'$\dot\theta$', [-0.08, 0.53], fontsize=18)
  plt.tick_params(direction='in')

  arrprop1 = {
    'arrowstyle': "Simple, tail_width=0.05, head_width=0.5, head_length=0.7",
    'connectionstyle': "arc3,rad=-0.2",
    'relpos': (1., 0.),
    'lw': 1.,
    'color': '#202020',
  }
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0.5,
    'alpha': 1
  }

  plt.title('Phase portrait of singular reduced dynamics', fontdict={'size': 13})

  plt.annotate(
    '',
    xy = (0.0 - 0.05, -dtheta_1 + 1),
    xytext = (0.5, 0.35),
    xycoords = 'data',
    textcoords = 'axes fraction',
    arrowprops = arrprop1
  )
  plt.annotate(
    '',
    xy = (theta_2 - 0.05, -dtheta_2 + 1),
    xytext = (0.5, 0.4),
    xycoords = 'data',
    textcoords = 'axes fraction',
    arrowprops = arrprop1
  )
  plt.annotate(
    'transition points',
    (0, 0),
    (0.20, 0.35),
    textcoords='axes fraction',
    xycoords='axes fraction',
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )

  arrprop2 = {
    'arrowstyle': "Simple, tail_width=0.05, head_width=0.5, head_length=0.7",
    'connectionstyle': "arc3,rad=0.2",
    'relpos': (1., 0.),
    'lw': 1.,
    'color': '#202020',
  }

  plt.annotate(
    '',
    xy = (0, -101),
    xytext = (0.3, -0.22),
    xycoords = 'data',
    textcoords = 'axes fraction',
    annotation_clip=False,
    arrowprops = arrprop2
  )
  plt.annotate(
    'node\nsingularity',
    (0, -100),
    (0.3, -0.22),
    xycoords='data',
    textcoords='axes fraction',
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )
  plt.annotate(
    '',
    xy = (theta_2, -101),
    xytext = (0.6, -0.22),
    xycoords = 'data',
    textcoords = 'axes fraction',
    annotation_clip=False,
    arrowprops = arrprop2
  )
  plt.annotate(
    'saddle\nsingularity',
    (theta_2, -100),
    (0.6, -0.22),
    xycoords='data',
    textcoords='axes fraction',
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )

  ax = axes[0]
  plt.sca(ax)
  plt.title('Pendubot: periodic motion near horizontal', fontdict={'size': 13})
  plt.grid(True, lw=0.5, ls='--')
  ax.set_aspect(1)
  plt.tick_params(direction='in')

  q_sing = data['singular_point']
  constr = data['constr']
  theta_1 = np.min(reduced_traj.coords)
  theta_3 = np.max(reduced_traj.coords)
  theta_2 = (theta_1 + theta_3) / 2
  q_1 = np.reshape(constr(theta_1), (2,))
  q_3 = np.reshape(constr(theta_3), (2,))
  q_2 = np.reshape(constr(theta_2), (2,))

  if False:
    x,y = get_vertix_coords(q_1, par)
    shadow1 = plt.plot(x, y, '-o', lw=6, markersize=12, color='#C0C0C0')
    line1 = plt.plot(x, y, '-o', lw=4, markersize=10, color='#6060C0')
    x,y = get_vertix_coords(q_2, par)
    shadow2 = plt.plot(x, y, '-o', lw=6, markersize=12, color='#C0C0C0')
    line2 = plt.plot(x, y, '-o', lw=4, markersize=10, color='#6060C0')
    x,y = get_vertix_coords(q_3, par)
    shadow3 = plt.plot(x, y, '-o', lw=6, markersize=12, color='#C0C0C0')
    line3 = plt.plot(x, y, '-o', lw=4, markersize=10, color='#6060C0')
  
  if True:
    view_par = DoublePendulumViewParam()
    view_par.joints_radius = [0.12, 0.12, 0.12]
    view_par.links_width = [0.08, 0.08]
    view_par.links_face = ('#A0A0F0',) * 2
    view1 = DoublePendulumView(view_par)
    view1.move(q_1)

    view2 = DoublePendulumView(view_par)
    view2.move(q_2)

    view3 = DoublePendulumView(view_par)
    view3.move(q_3)

    for p in view1.patches + view2.patches + view3.patches:
      ax.add_patch(p)

  plt.xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2], [])
  plt.yticks([-1, -0.5, 0, 0.5, 1, 1.5, 2], [])
  plt.xlim(-0.08, 1.85)
  plt.ylim(-1.2, 0.2)

  plt.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
  plt.show()
  # plt.savefig('fig/pendubot-preview.pdf', dpi=300)

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  make_preview()
