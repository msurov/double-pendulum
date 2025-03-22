import numpy as np
from common.trajectory import (
  Trajectory,
  traj_join, 
  traj_forth_and_back, 
  traj_repeat
)
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam
)
import casadi as ca
import matplotlib.pyplot as plt
from double_pendulum.anim import draw, animate
from double_pendulum.motion_planner.singular_constrs import get_sing_constr_at
from double_pendulum.motion_planner.reduced_dynamics import (
  ReducedDynamics,
  solve_reduced,
  compute_time,
  reconstruct_trajectory
)
import scienceplots
from fractions import Fraction
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter


def get_test_par() -> DoublePendulumParam:
  return DoublePendulumParam(
    lengths=[1., 1.],
    mass_centers=[0.5, 0.5],
    masses=[0.2, 0.2],
    inertia=[0.05, 0.05],
    actuated_joint=0,
    gravity_accel=9.81
  )

def get_aux_par(par : DoublePendulumParam):
  I1,I2 = par.inertia
  c1,c2 = par.mass_centers
  m1,m2 = par.masses
  l1,l2 = par.lengths
  g = par.gravity_accel

  p1 = I1 + I2 + c1**2 * m1 + c2**2 * m2 + l1**2 * m2
  p2 = m2 * c2 * l1
  p3 = I2 + m2 * c2**2
  p4 = c1 * m1 + l1 * m2
  p5 = c2 * m2

  return [p1, p2, p3, p4, p5]

constr1 = [
  [-2.045583727546234, 0.6462469356355233],
  [1.393523066250672, -2.506041495691292],
  [-0.1363301878405378, -1.314775411393767]
]
constr2 = [
  [-1.7050253662106756, -3.30435292667277],
  [1.6212977144817882, -0.021427457624774108],
  [0.27257839179537074, -0.2201992662211332]
]
constr3 = [
  [11.690270453301533, 2.0109207709898262],
  [1.9875242619058198, -1.1407354700119394],
  [-2.3846017025850132, -1.6691408523125282]
]
constr4 = [
  [1.0, -2.5],
  [-1.6937072630535512, 0.33680450265272777],
  [0.5341867462328763, 0.8089709547202198]
]
constr5 = [
  [2.5, 1.3],
  [0.7209723051505261, -0.9138315522490605],
  [-0.2838704130414532, -0.22401842006986564]
]
constr6 = [
  [-1, 2.5708],
  [ 4.36314, -0.691685],
  [-3.40499, -1.746941],
]
constr7 = [
  [-1, 2.4],
  [4.07125, -1.06913],
  [3.41457, -5.54724],
]
constr8 = [
  [-1, 2.7],
  [4.58158, -0.439501],
  [-1.04110654990786, -1.090042343381951],
]
constr9 = [
  [-2, 2.8],
  [4.73436, -0.27354],
  [-0.10471036876355884, -1.75698659010537],
]
constr10 = [
  [-0.5, 2.5],
  [4.24058, -0.843266],
  [-1.6285926672387678, -4.03333536861941],
]
constr11 = [
  [-1, 0.6],
  [4.31254, -7.87183],
  [-6.194525037457379, -3.945033830958353],
]
constr12 = [
  [-1.2, 0.8],
  [3.97678, -6.74742],
  [-7.286772954585941, -3.3208549438556756],
]
constr13 = [
  [-2, 0.9],
  [3.82614, -6.2045],
  [-6.366655328484425, -4.971025026051284],
]
constr14 = [
  [-2.5, 2.8],
  [4.73436, -0.27354],
  [-0.1794428230674952, -2.856044538839678],
]
constr15 = [
  [ 2.35619449, -0.78539816],
  [ 4.        , -6.82842712],
  [ 6.91714436,  4.05196935]
]
constr16 = [
  [-1.04719755,  2.61799388],
  [ 4.44444444, -0.59544265],
  [-0.28296041, -5]
]

def get_constr(isample : int) -> ca.Function:
  s = ca.SX.sym('s')
  constr = eval(f'constr{isample}')
  expr = ca.DM(constr[0]) \
    + ca.DM(constr[1]) * s \
    + ca.DM(constr[2]) * s**2 / 2
  return ca.Function('constr', [s], [expr])

def enlarge_rect(r, coef):
  c = np.mean(r, axis=1)
  w = r[:,1] - r[:,0]
  return np.array([c - w * coef / 2, c + w * coef / 2]).T

def get_traj_bounding_rect(traj : Trajectory):
  qmin = np.min(traj.coords, axis=0)
  qmax = np.max(traj.coords, axis=0)
  return np.array([qmin, qmax]).T

def get_cartesian_rect(traj : Trajectory, par : DoublePendulumParam):
  q1, q2 = traj.coords.T
  x1 = par.lengths[0] * np.sin(q1)
  y1 = par.lengths[0] * np.cos(q1)
  x2 = x1 + par.lengths[0] * np.sin(q1 + q2)
  y2 = y1 + par.lengths[0] * np.cos(q1 + q2)

  xmin = min(0, np.min(x1))
  xmin = min(xmin, np.min(x2))

  xmax = max(0, np.max(x1))
  xmax = max(xmax, np.max(x2))

  ymin = min(0, np.min(y1))
  ymin = min(ymin, np.min(y2))

  ymax = max(0, np.max(y1))
  ymax = max(ymax, np.max(y2))

  return np.array([
    [xmin, xmax],
    [ymin, ymax]
  ])

def motion_schematic(traj : Trajectory, par : DoublePendulumParam):
  d = traj.phase - traj.phase[0]
  d = np.linalg.norm(d, axis=1)
  i, = np.nonzero(d < 1e-5)
  i = i[1]
  q1 = traj.coords[0]
  q2 = traj.coords[i//4]
  q3 = traj.coords[i//2]

  with plt.style.context('science'):
    fig,ax = plt.subplots(1, 1, num='schematic', figsize=(6, 4))
    ax.set_aspect(1)
    draw(q1, par, alpha=1, color='#3030E0', linewidth=2)
    draw(q2, par, alpha=1, color='#3030C0', linewidth=2)
    draw(q3, par, alpha=1, color='#3030A0', linewidth=2)
    r = get_cartesian_rect(traj, par)
    xdiap, ydiap = enlarge_rect(r, 1.05)
    ax.set_xlim(*xdiap)
    ax.set_ylim(*ydiap)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/horizontal_oscillations_schematic.pdf')

def add_annotation(text : str, textpos : Tuple[int, int]):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 0.8
  }
  annotate_par = {
    'xycoords': 'axes points',
    'font': {
      'size': 18
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def show_ref_traj(traj : Trajectory):
  with plt.style.context('science'):
    _,axes = plt.subplots(2, 2, sharex=True, num='phase trajectory projections', figsize=(6, 4))
    theta1, theta2, dtheta1, dtheta2 = traj.phase.T

    print(np.min(theta1), np.max(theta1))
    print(np.min(theta2), np.max(theta2))

    u = traj.control
    plt.sca(axes[0, 0])
    # theta1_ticks, theta1_labels = gen_pi_ticks('29/40', '31/40', '1/40')
    # plt.xticks(theta1_ticks, theta1_labels)
    # theta2_ticks, theta2_labels = gen_pi_ticks('-7/24', '-5/24', '1/24')
    # plt.yticks(theta2_ticks, theta2_labels)
    plt.grid(True)
    plt.plot(theta1, theta2, alpha=0.8)
    add_annotation(R'$q_2$', (8, 75))

    plt.sca(axes[0,1])
    plt.grid(True)
    plt.plot(theta1, dtheta1, alpha=0.8)
    add_annotation(R'$\dot q_1$', (8, 92))

    plt.sca(axes[1, 0])
    plt.grid(True)
    plt.plot(theta1, u, alpha=0.8)
    add_annotation(R'$q_1$', (70, -20))
    add_annotation(R'$u$', (16, 92))

    plt.sca(axes[1, 1])
    plt.grid(True)
    plt.plot(theta1, dtheta2, alpha=0.8)
    add_annotation(R'$q_1$', (70, -20))
    add_annotation(R'$\dot q_2$', (8, 92))

    plt.tight_layout(h_pad=-0., w_pad=0.2)
    plt.savefig('data/horizontal_oscillations_trajectory.pdf')


def show_phase_prortrait(reduced : ReducedDynamics, reduced_traj : Trajectory):
  sleft = np.min(reduced_traj.coords)
  sright = np.max(reduced_traj.coords)
  dsmin = np.min(reduced_traj.vels)
  dsmax = np.max(reduced_traj.vels)

  with plt.style.context('science'):
    plt.figure('phase', figsize=(6, 4))
    plt.axhline(0, color='black', alpha=0.5, lw=1)
    plt.axvline(0, color='black', alpha=0.5, lw=1)

    s1 = sleft * 1.2
    s2 = sright * 1.2
    ds1 = dsmin * 1.2
    ds2 = dsmax * 1.2
    s = np.linspace(s1, s2, 30)
    ds = np.linspace(ds1, ds2, 30)
    X,Y = np.meshgrid(s, ds)
    U = np.zeros(X.shape)
    V = np.zeros(X.shape)
    for i in range(len(s)):
      for j in range(len(ds)):
        U[j,i] = ds[j]
        V[j,i] = (-reduced.beta(s[i]) * ds[j]**2 - reduced.gamma(s[i])) / reduced.alpha(s[i])

    plt.streamplot(X, Y, U, V, color='lightblue')
    plt.plot(reduced_traj.coords, reduced_traj.vels, lw=2, color='darkblue', alpha=1)
    plt.gca().set_xlim(s1, s2)
    plt.gca().set_ylim(ds1, ds2)
    add_annotation(R'$\theta$', (340, 10))
    add_annotation(R'$\dot\theta$', (8, 210))

    plt.tight_layout()
    plt.savefig('data/horizontal_oscillations_phase.pdf')

def show_aux_par(par):
  p = get_aux_par(par)
  for i in range(len(p)):
    print(f'p_{i + 1} = {p[i]}')

def old(isample):
  par = get_test_par()
  print(par)
  show_aux_par(par)
  dynamics = DoublePendulumDynamics(par)
  constr = get_constr(isample)
  reduced = ReducedDynamics(dynamics, constr)
  tr_left = solve_reduced(reduced, [-0.02, -1e-4], 0.0, max_step=1e-4)
  tr_right = solve_reduced(reduced, [0.02, 1e-4], 0.0, max_step=1e-4)
  tr_up = traj_join(tr_left, tr_right[::-1])
  tr_closed = traj_forth_and_back(tr_up)
  tr_reduced = traj_repeat(tr_closed, 2)
  tr_orig = reconstruct_trajectory(constr, reduced, dynamics, tr_reduced)

  show_phase_prortrait(reduced, tr_closed)

  show_ref_traj(tr_orig)

  motion_schematic(tr_orig, par)
  plt.show()

  a = animate(tr_orig, par, speedup=0.1)
  plt.tight_layout()
  plt.grid(True)
  a.save(f'data/{isample}.mp4')
  plt.show()

  # _,axes = plt.subplots(3, 1, sharex=True)
  # s = np.linspace(-0.1, 0.1)
  # alpha = [float(reduced.alpha(e)) for e in s]
  # beta = [float(reduced.beta(e)) for e in s]
  # gamma = [float(reduced.gamma(e)) for e in s]
  # plt.sca(axes[0])
  # plt.grid(True)
  # plt.axhline(0, color='black', lw=1)
  # plt.plot(s, alpha)
  # plt.sca(axes[1])
  # plt.grid(True)
  # plt.axhline(0, color='black', lw=1)
  # plt.plot(s, beta)
  # plt.sca(axes[2])
  # plt.grid(True)
  # plt.axhline(0, color='black', lw=1)
  # plt.plot(s, gamma)
  # plt.tight_layout()
  # plt.savefig('data/alpha-beta-gamma.pdf')

  # plt.figure('phase')
  # plt.axhline(0, color='black', lw=1)
  # plt.axvline(0, color='black', lw=1)
  # plt.grid(True)
  # t,s,ds = solve_reduced(reduced, [-0.09, -0.0001], 0.01, max_step=1e-4)
  # plt.plot(s, ds, color=color, lw=2)
  # plt.plot(s, -ds, color=color, lw=2)
  # t,s,ds = solve_reduced(reduced, [0.09, 0.0001], 0.01, max_step=1e-4)
  # plt.plot(s, ds, color=color, lw=2)
  # plt.plot(s, -ds, color=color, lw=2)
  # plt.savefig('data/phase.pdf')
  # plt.show()

  # plt.figure()
  # plt.grid(True)
  # q = np.reshape(constr(0.), (-1,))
  # draw(q, par, alpha=1.)
  # q = np.reshape(constr(-0.1), (-1,))
  # draw(q, par, alpha=0.5)
  # q = np.reshape(constr(0.1), (-1,))
  # draw(q, par, alpha=0.5)
  # plt.savefig('data/configuration.pdf')
  # plt.show()

def gen_pi_ticks(xfrom : Fraction, xto : Fraction, xstep : Fraction) -> Tuple[List[float], List[str]]:
  xfrom = Fraction(xfrom)
  xto = Fraction(xto)
  xstep = Fraction(xstep)

  assert xstep > 0
  assert xto > xfrom

  ticks = []
  labels = []
  x = xfrom

  while x <= xto:
    if x.numerator == 0:
      s = '$0$'
    elif x == 1:
      s = '$\pi$'
    elif x.denominator == 1:
      s = R'$' + str(x.numerator) + R'\pi$'
    elif x < 0:
      s = R'$-\frac{' + str(-x.numerator) + R'}{' + str(x.denominator) + R'}\pi$'
    else:
      s = R'$\frac{' + str(x.numerator) + R'}{' + str(x.denominator) + R'}\pi$'
    ticks.append(np.pi * x)
    labels.append(s)
    x += xstep
  
  return ticks, labels

def show_oscillatory_points():
  plt.figure('configuration space', figsize=(4, 4))
  color = 'lightblue'

  # 0..pi/2
  q2 = np.linspace(0, np.pi/2)
  q1_min = np.pi - q2
  q1_max = 2*np.pi - q2
  plt.fill_between(q2, q1_min, q1_max, color=color)

  # pi/2..pi
  q2 = np.linspace(np.pi/2, np.pi)
  q1_min = 2*np.pi - q2
  q1_max = 3*np.pi - q2
  plt.fill_between(q2, q1_min, q1_max, color=color)

  q2 = np.linspace(np.pi/2, np.pi)
  q1_min = 0*np.pi - q2
  q1_max = np.pi - q2
  plt.fill_between(q2, q1_min, q1_max, color=color)

  # pi..3pi/2
  q2 = np.linspace(np.pi, 3*np.pi/2)
  q1_min = np.pi - q2
  q1_max = 2*np.pi - q2
  plt.fill_between(q2, q1_min, q1_max, color=color)

  q2 = np.linspace(np.pi, 3*np.pi/2)
  q1_min = 3*np.pi - q2
  q1_max = 4*np.pi - q2
  plt.fill_between(q2, q1_min, q1_max, color=color)

  # 3pi/2..2pi
  q2 = np.linspace(3*np.pi/2, 2*np.pi)
  q1_min = 2*np.pi - q2
  q1_max = 3*np.pi - q2
  h2 = plt.fill_between(q2, q1_min, q1_max, color=color)

  q2 = np.linspace(0, 2*np.pi)
  q1 = -q2
  color = 'brown'
  h1, = plt.plot(q1, q2, color=color)
  q1 = np.pi - q2
  plt.plot(q1, q2, color=color)
  q1 = 2*np.pi - q2
  plt.plot(q1, q2, color=color)
  q1 = 3*np.pi - q2
  plt.plot(q1, q2, color=color)

  ax = plt.gca()
  ax.set_aspect(1)
  plt.xlabel('$q_2$')
  plt.ylabel('$q_1$')
  plt.xlim(0, 2*np.pi)
  plt.ylim(0, 2*np.pi)

  t, l = gen_pi_ticks(0, 2, '1/2')
  ax.set_xticks(t, l)
  ax.set_yticks(t, l)

  plt.grid(True)
  plt.tight_layout()
  plt.savefig('data/configurations_with_oscillations.pdf')
  plt.show()


if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.size": 14,
      "font.family": "Helvetica"
  })

  np.set_printoptions(suppress=True)
  main(isample=16)
  # show_oscillatory_points()