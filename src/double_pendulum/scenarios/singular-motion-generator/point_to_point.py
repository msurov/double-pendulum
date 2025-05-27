import numpy as np
import matplotlib.pyplot as plt
from common.mechsys import MechanicalSystem
from common.plots import set_pi_xticks
from common.numpy_utils import integrate_array, cont_angle, map_array, normalized
import casadi as ca
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default,
  convert_parameters
)
from singular_motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from double_pendulum.scenarios.singular_oscillations_planner import (
  show_trajectory_projections,
  show_reduced_dynamics_phase_prortrait,
)
from double_pendulum.scenarios.transverse_feedback_closed_loop_sim import (
  DoublePendulumTransverseFeedback,
  TranverseFeedbackController,
  TransverseDynamics,
  TranverseFeedbackControllerPar,
  TransverseDynamics,
)

from double_pendulum.anim import (
  motion_schematic,
  motion_schematic_v2
)
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches



q_sing = np.array([-0.5, 1.8])
par = DoublePendulumParam(
  [1, 1],
  [0.5, 0.5],
  [1, 1],
  [0, 0],
  0,
  1
)
amplitude = 0.5

def get_sing_constr(dynamics : MechanicalSystem):
  B = dynamics.B_expr
  B_perp = dynamics.B_perp_expr
  G = dynamics.G_expr

  gam = ca.evalf(ca.substitute(B_perp @ G, dynamics.q, q_sing))
  if abs(gam) < 1e-6:
    return None

  if gam < 0:
    B_perp = -B_perp

  q = dynamics.q
  dq = dynamics.dq
  M = dynamics.M_expr
  C = dynamics.C_expr

  N = ca.solve(M, B)
  P = M @ B_perp.T
  F = B_perp.T / (B_perp @ M @ B_perp.T)

  left = B_perp @ C @ N
  left = ca.substitute(left, dq, N)
  left = ca.substitute(left, q, q_sing)
  left = ca.evalf(left)
  right = N.T @ ca.jtimes(P, dynamics.q, N)
  right = ca.substitute(right, q, q_sing)
  right = ca.evalf(right)

  if left >= right:
    return None

  N_val = ca.evalf(ca.substitute(N, q, q_sing))
  F_val = ca.evalf(ca.substitute(F, q, q_sing))

  theta = ca.SX.sym('theta')
  constr_expr = q_sing + 0.8 * N_val * theta - 0.08 * F_val * theta**2
  constr_fun = ca.Function('constr', [theta], [constr_expr])

  constr_deriv_expr = ca.jacobian(constr_expr, theta)
  constr_deriv_fun = ca.Function('constr_deriv', [theta], [constr_deriv_expr])

  return constr_fun, constr_deriv_fun

def plot_arrow(ax, p, v, color, **kwargs):
  p = np.reshape(p, (2,))
  v = np.reshape(v, (2,))
  arrow = FancyArrowPatch(
    p,
    p + v,
    connectionstyle="arc3,rad=0.0",
    arrowstyle="Simple, tail_width=1, head_width=7, head_length=10",
    color=color,
    **kwargs
  )
  ax.add_patch(arrow)
  return arrow

class FancyArrowPatchHandler(HandlerPatch):
  def create_artists(self, legend, orig_handle,
                     xdescent, ydescent, width, height, fontsize, trans):
    center = np.array([
      0.1 * width - 0.5 * xdescent,
      0.5 * height - 0.5 * ydescent
    ])
    p = FancyArrowPatch(
      center,
      center + np.array([0.8 * width, 0.]),
      connectionstyle="arc3,rad=0.0",
      arrowstyle="Simple, tail_width=1, head_width=7, head_length=10",
    )
    self.update_prop(p, orig_handle, legend)
    p.set_transform(trans)
    return [p]

def add_textbox(ax, text, text_pos_data=None, text_pos_ratio=None, offsetpts=(0, 0), fontsize=18, **kwargs):
  bbox = {
        'boxstyle': 'round',
        'fc': 'white',
        'ec': 'white',
        'alpha': 0.7,
      }
  if text_pos_data is not None:
    text_pos = text_pos_data
    xycoords='data'
  elif text_pos_ratio is not None:
    text_pos = text_pos_ratio
    xycoords='axes fraction'
  else:
    assert False

  ax.annotate(
    text, xy=text_pos, xytext=offsetpts,
    ha='left', va='top', 
    xycoords=xycoords,
    textcoords='offset points',
    fontsize=fontsize,
    bbox = bbox,
    **kwargs
  )

def main():
  dynamics = DoublePendulumDynamics(par)
  constr, constr_deriv = get_sing_constr(dynamics)

  theta = np.linspace(-amplitude, amplitude, 100)
  traj = map_array(constr, theta, (2,))
  q_start = traj[0]
  q_end = traj[-1]

  q_min = np.min(traj, axis=0)
  q_max = np.max(traj, axis=0)

  arrow_length = 0.2
  d = q_max - q_min
  q1_range = np.array([q_min[0] - 0.1, q_max[0] + 0.3])
  q2_range = np.array([q_min[1] - 0.2, q_max[1] + 0.15])
  ratio = (q1_range[1] - q1_range[0]) / (q2_range[1] - q2_range[0])

  vec_field_expr = dynamics.B_perp_expr @ dynamics.M_expr
  vec_field_fun = ca.Function('vec_field', [dynamics.q], [vec_field_expr])

  q1 = np.linspace(*q1_range, 30)
  q2 = np.linspace(*q2_range, int(len(q1) / ratio + 0.5))
  q1_mesh, q2_mesh = np.meshgrid(q1, q2)
  vx_mesh = np.zeros(q1_mesh.shape)
  vy_mesh = np.zeros(q1_mesh.shape)

  for i2 in range(q1_mesh.shape[0]):
    for i1 in range(q1_mesh.shape[1]):
      v = vec_field_fun([q1_mesh[i2,i1], q2_mesh[i2,i1]])
      vx_mesh[i2,i1] = float(v[0])
      vy_mesh[i2,i1] = float(v[1])

  fig, ax = plt.subplots(1, 1)
  ax.set_aspect(1)
  plt.streamplot(q1_mesh, q2_mesh, vx_mesh, vy_mesh, density=0.8, color='#B0B0B0')

  plt.plot(traj[:,0], traj[:,1], color='black', lw=4, label='$q(t)$')

  tan_arrow_color = '#6060F0'
  perp_arrow_color = '#F06060'

  v_start = arrow_length * normalized(constr_deriv(-amplitude))
  w_start = arrow_length * normalized(vec_field_fun(q_start))
  plot_arrow(ax, q_start, v_start, tan_arrow_color, zorder=10, lw=2)
  plot_arrow(ax, q_start, w_start, perp_arrow_color, zorder=10, lw=2)
  plt.plot(q_start[0], q_start[1], 'o', color='black', markersize=8, zorder=20)

  v_end = arrow_length * normalized(constr_deriv(amplitude))
  w_end = arrow_length * normalized(vec_field_fun(q_end))
  plot_arrow(ax, q_end, v_end, tan_arrow_color, zorder=10, lw=2)
  plot_arrow(ax, q_end, w_end, perp_arrow_color, zorder=10, lw=2)
  plt.plot(q_end[0], q_end[1], 'o', color='black', markersize=8, zorder=20)

  v_sing = arrow_length * normalized(constr_deriv(0))
  w_sing = arrow_length * normalized(vec_field_fun(q_sing))
  plot_arrow(ax, q_sing, v_sing, tan_arrow_color, zorder=10, label='$\dot q$', lw=2)
  plot_arrow(ax, q_sing, w_sing, perp_arrow_color, zorder=10, label=R'$B^\perp(q) M(q)$', lw=2)
  plt.plot(q_sing[0], q_sing[1], 'o', color='black', markersize=8, zorder=20)

  add_textbox(ax, R'$q_{\mathrm{start}}$', text_pos_data=q_start, 
              offsetpts=(5, 35), fontsize=27, color='#202020')
  add_textbox(ax, R'$q_{\mathrm{sing}}$', text_pos_data=q_sing, 
              offsetpts=(5, 35), fontsize=27, color='#202020')
  add_textbox(ax, R'$q_{\mathrm{end}}$', text_pos_data=q_end, 
              offsetpts=(5, 35), fontsize=27, color='#202020')

  plt.legend(
    fontsize=18, loc='upper right', 
    handler_map={FancyArrowPatch: FancyArrowPatchHandler(ax)}
  )

  plt.xlim(*q1_range)
  plt.ylim(*q2_range)

  plt.xticks([])
  plt.yticks([])

  add_textbox(ax, R'configuration space', text_pos_ratio=(0.03, 0.08), 
              fontsize=18, color='#202020')

  plt.tight_layout(pad=0.1)
  plt.savefig('fig/singularity-condition.pdf')
  return fig


def show_traj():
  dynamics = DoublePendulumDynamics(par)
  constr, _ = get_sing_constr(dynamics)
  reduced = ReducedDynamics(dynamics, constr)

  dtheta_sing = float(np.sqrt(-reduced.gamma(0.) / reduced.beta(0.)))
  tr_left = solve_reduced(reduced, [-1e-3, -amplitude], dtheta_sing, max_step=1e-3)
  tr_right = solve_reduced(reduced, [1e-3, amplitude], dtheta_sing, max_step=1e-3)
  tr_reduced = traj_join(tr_left[::-1], tr_right)
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)

  theta = np.linspace(-amplitude, amplitude)
  alpha = map_array(reduced.alpha, theta, elem_size=(1,))
  plt.figure('alpha')
  plt.plot(theta, alpha)

  plt.figure('phase')
  plt.plot(tr_orig.coords[:,0], tr_orig.coords[:,1])

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.size": 14,
      "font.family": "Helvetica"
  })
  main()
  # show_traj()
  plt.show()
