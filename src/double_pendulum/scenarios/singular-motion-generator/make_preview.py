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
from common.numpy_utils import map_array
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
  qs = ca.DM([-ca.pi/4, 3*ca.pi/4])
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
  eps = 1e-2

  # verify_reduced_dynamics(par, reduced)
  tr_1 = solve_reduced(reduced, [-3.9, -eps/2], 0.0, max_step=eps)
  tr_2 = solve_reduced(reduced, [3.8, eps/2], 0.0, max_step=eps)[::-1]
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

  if False:
    theta = np.linspace(-10, 10)
    alpha = map_array(reduced_dynamics.alpha, theta, 1)
    beta = map_array(reduced_dynamics.beta, theta, 1)
    gamma = map_array(reduced_dynamics.gamma, theta, 1)

    plt.plot(theta, alpha * 1000)
    plt.plot(theta, beta * 1000)
    plt.plot(theta, gamma)

    plt.axhline(0, color='black')
    plt.grid(True)
    plt.show()
    exit()

  fig, axes = plt.subplots(1, 2, figsize=(8, 4), width_ratios=(0.85, 1))

  theta_min = -12
  saddle_sing = brentq(reduced_dynamics.alpha, -12, -5)
  node_sing = 0.
  theta_max = 6.
  node_speed = np.sqrt(-reduced_dynamics.gamma(node_sing) / reduced_dynamics.beta(node_sing))
  node_speed = float(node_speed)
  step = 0.5
  eps = 1e-2
  speed_max = 100

  saddle_speed = np.sqrt(-reduced_dynamics.gamma(saddle_sing) / reduced_dynamics.beta(saddle_sing))
  saddle_speed = float(saddle_speed)

  ax = axes[1]
  plt.sca(ax)
  plt.grid(True, lw=0.5, ls='--')
  plt.axvline(saddle_sing, ls='--', lw=1, color='#404040')
  plt.axhline(0, ls='-', lw=1, color='#404040')
  plt.axvline(node_sing, ls='--', lw=1, color='#404040')

  if True:
    for x0 in np.linspace(saddle_sing - step, theta_min, 4, endpoint=False):
      show_phase_curve(reduced_dynamics, [x0, theta_min], eps)
      show_phase_curve(reduced_dynamics, [x0, theta_min], speed_max)

    for x0 in np.linspace(saddle_sing + step, node_sing - step, 10):
      show_phase_curve(reduced_dynamics, [x0, node_sing - eps], eps)
      show_phase_curve(reduced_dynamics, [x0, node_sing - eps], speed_max)

    for x0 in np.linspace(node_sing + step, theta_max, 6):
      show_phase_curve(reduced_dynamics, [x0, node_sing + eps], eps)
      show_phase_curve(reduced_dynamics, [x0, node_sing + eps], speed_max)

    for dx0 in np.linspace(speed_max/3, speed_max, 4, endpoint=False):
      show_phase_curve(reduced_dynamics, [theta_max, node_sing - eps], dx0)

    show_phase_curve(reduced_dynamics, [saddle_sing + eps, node_sing - eps], saddle_speed)
    show_phase_curve(reduced_dynamics, [saddle_sing - eps, theta_min], saddle_speed)

  plt.plot(reduced_traj.coords, reduced_traj.vels, lw=2, color='#C04040')
  plt.xlim(theta_min, theta_max)
  plt.ylim(-speed_max, speed_max)
  plt.yticks(np.linspace(-speed_max, speed_max, 7), [])

  step = abs(saddle_sing - node_sing) / 3
  plt.xticks(np.arange(saddle_sing, theta_max, step), [])
  add_annotation(R'$\theta$', [0.92, 0.05], fontsize=20)
  add_annotation(R'$\dot\theta$', [0.03, 0.88], fontsize=20)
  plt.tick_params(direction='in')

  arrprop1 = {
    'arrowstyle': "Simple, tail_width=0.05, head_width=0.5, head_length=0.7",
    'connectionstyle': "arc3,rad=-0.2",
    'relpos': (1., 0.),
    'lw': 1.,
    'color': '#202020',
  }

  arrprop2 = {
    'arrowstyle': "Simple, tail_width=0.05, head_width=0.5, head_length=0.7",
    'connectionstyle': "arc3,rad=0.2",
    'relpos': (1., 0.),
    'lw': 1.,
    'color': '#202020',
  }

  bbox = {
    'boxstyle': 'round',
    'ec': '#A060A0',
    'fc': '1.0',
    'lw': 0.5,
    'alpha': 0.8
  }

  plt.title('Phase portrait of singular reduced dynamics', fontdict={'size': 13})

  plt.annotate(
    '',
    xy = (saddle_sing + 0.2, -saddle_speed - 1),
    xytext = (0.33, 0.15),
    xycoords = 'data',
    textcoords = 'axes fraction',
    arrowprops = arrprop1
  )
  plt.annotate(
    '',
    xy = (node_sing - 0.2, -node_speed - 1),
    xytext = (0.54, 0.15),
    xycoords = 'data',
    textcoords = 'axes fraction',
    arrowprops = arrprop2
  )
  plt.annotate(
    'transition\npoints',
    (0, 0),
    (0.35, 0.06),
    textcoords='axes fraction',
    xycoords='axes fraction',
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 13},
  )

  plt.annotate(
    'node\nsingularity',
    (0.6, -0.15),
    (0.6, -0.15),
    xycoords='axes fraction',
    textcoords='axes fraction',
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )
  plt.annotate(
    'saddle\nsingularity',
    (0.1, -0.15),
    (0.1, -0.15),
    xycoords='axes fraction',
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

    view_par.links_face = ('#D0D0FF',) * 2
    view_par.links_edge = ('#A0A0A0',) * 2
    view_par.joints_face = ('#C0C0C0',) * 3
    view_par.joints_edge = ('#A0A0A0',) * 3
    view1 = DoublePendulumView(view_par)
    view1.move(q_3)

    view_par.links_face = ('#A0A0F0',) * 2
    view_par.links_edge = ('#808080',) * 2
    view_par.joints_face = ('#A0A0A0',) * 3
    view_par.joints_edge = ('#606060',) * 3
    view2 = DoublePendulumView(view_par)
    view2.move(q_2)

    view_par.links_face = ('#D0D0FF',) * 2
    view_par.links_edge = ('#A0A0A0',) * 2
    view_par.joints_face = ('#C0C0C0',) * 3
    view_par.joints_edge = ('#A0A0A0',) * 3
    view3 = DoublePendulumView(view_par)
    view3.move(q_1)

    for p in view1.patches + view3.patches + view2.patches:
      ax.add_patch(p, )

  plt.xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2], [])
  plt.yticks([-1, -0.5, 0, 0.5, 1, 1.5, 2], [])
  plt.xlim(-1.05, 0.8)
  plt.ylim(-0.3, 1.4)

  plt.tight_layout(pad=1, h_pad=0.1, w_pad=0.1)
  plt.savefig('fig/pendubot-preview.pdf', dpi=300)
  plt.show()

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  make_preview()
