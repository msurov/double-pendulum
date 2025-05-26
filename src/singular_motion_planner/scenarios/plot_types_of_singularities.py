import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple
from scipy.integrate import solve_ivp


ReducedDynamicsRHS = Callable[[float, float], float]

def show_phase(reduced_dynamics : ReducedDynamicsRHS, x_diap, dx0):
  def rhs(x, y):
    return reduced_dynamics(x, y) / y

  def stop_cond(x, y):
    if y[0] <= 0:
      return 0
    return 1
  stop_cond.terminal = True

  sol = solve_ivp(rhs, x_diap, [dx0], max_step=1e-2, events=stop_cond)
  plt.plot(sol.t, sol.y[0], color='#8080C0', alpha=0.8, lw=1)
  plt.plot(sol.t, -sol.y[0], color='#8080C0', alpha=0.8, lw=1)


def add_annotation(text : str, textpos : Tuple[int, int], fontsize=16):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0,
    'alpha': 1
  }
  annotate_par = {
    'xycoords': 'axes fraction',
    'font': {
      'size': fontsize
    },
    'bbox': bbox
  }
  return plt.annotate(text, textpos, **annotate_par)

def demo_isolated_sing():
  add_annotation(R'$\theta \, \ddot{\theta}-\dot{\theta}^{2}+1=0$', [0.53, 0.90])
  # add_annotation(R'(a) isolated center singularity', [0.05, -0.2], 14)
  add_annotation('(A)', [0.03, 0.05], 15)

  step = 1e-3
  sys1 = lambda x, dx: (dx**2 - 1.0) / x

  # left to sing
  for x0 in np.linspace(-0.5, -0.05, 4):
    show_phase(sys1, [x0, -step/2], 1e-3)
    show_phase(sys1, [x0, -step/2], 2.)

  # right to ting
  for x0 in np.linspace(0.05, 1.5, 12):
    show_phase(sys1, [x0, step/2], 1e-3)
    show_phase(sys1, [x0, step/2], 2.)

  # from sides
  for dx0 in np.linspace(0.5, 1.7, 6):
    show_phase(sys1, [-0.5, -step/2], dx0)

  for dx0 in np.linspace(0.5, 1.7, 6):
    show_phase(sys1, [1.5, step/2], dx0)

  plt.xlim(-0.5, 1.5)
  plt.ylim(-2., 2.)

  plt.axvline(0, color='black', lw=1, zorder=-1)
  plt.axhline(0, color='black', lw=1, zorder=-1)

  plt.yticks([-1, 0, 1], fontsize=13)
  plt.xticks([0., 0.5, 1], fontsize=13)
  plt.tick_params(direction='in')

def demo_saddle_equil():
  add_annotation(R'$\theta \, \ddot{\theta}-\dot{\theta}^{2}+\cos{\frac{\pi \theta}{2}}=0$', [0.42, 0.90])
  # add_annotation(R'(b) saddle equilibrium', [0.05, -0.2], 14)
  add_annotation('(B)', [0.03, 0.05], 15)

  step = 1e-3
  sys1 = lambda x, dx: (dx**2 - np.cos(np.pi*x/2)) / x

  # left to sing
  for x0 in np.arange(-0.1, -0.6, -0.15):
    show_phase(sys1, [x0, -step/2], 1e-3)
    show_phase(sys1, [x0, -step/2], 2.)

  for dx0 in np.arange(0.2, 2., 0.4):
    show_phase(sys1, [-0.6, -step/2], dx0)

  # between sing and saddle
  for x0 in np.linspace(1 - step, 0.1, 8):
    show_phase(sys1, [x0, step/2], 1e-3)
    show_phase(sys1, [x0, step/2], 2.0)

  # right to saddle
  for x0 in np.arange(1+step, 1.65, 0.15):
    show_phase(sys1, [x0, 2.], 1e-3)

  for x0 in np.arange(1+step, 1.65, 0.15):
    show_phase(sys1, [x0, step/2], 2.)

  for dx0 in np.arange(0.85, 2., 0.2):
    show_phase(sys1, [1.65, step], dx0)

  plt.axvline(0, color='black', lw=1, zorder=-1)
  plt.axvline(1, color='grey', ls='--', lw=1, zorder=-1)
  plt.axhline(0, color='black', lw=1, zorder=-1)
  plt.xlim(-0.5, 1.5)
  plt.ylim(-2., 2.)
  plt.yticks([-1, 0, 1], fontsize=13)
  plt.xticks([0., 0.5, 1], fontsize=13)
  plt.tick_params(direction='in')

def demo_saddle_sing():
  add_annotation(R'$\frac{\sin\pi \theta}{\pi} \, \ddot{\theta}-\dot{\theta}^{2}+1=0$', [0.45, 0.90])
  # add_annotation(R'(c) saddle singularity', [0.05, -0.2], 14)
  add_annotation('(C)', [0.03, 0.05], 15)

  step = 1e-3
  sys1 = lambda x, dx: (dx**2 - 1) * np.pi / np.sin(np.pi*x)

  # left to sing1
  for x0 in np.linspace(-0.5, -0.1, 6):
    show_phase(sys1, [x0, -step/2], 1e-3)
    show_phase(sys1, [x0, -step/2], 2.)

  for dx0 in np.linspace(0.5, 1.7, 6):
    show_phase(sys1, [-0.5, -step/2], dx0)

  # between sings
  for x0 in np.linspace(0.1, 1 - 0.05, 8):
    show_phase(sys1, [x0, step/2], 1e-3)
    show_phase(sys1, [x0, step/2], 2.)

  for x0 in np.linspace(1 + 0.05, 1.5, 8):
    show_phase(sys1, [x0, 1.5], 1e-3)
    show_phase(sys1, [x0, 1.5], 2.)

  show_phase(sys1, [1.5, 1+step], 1.)
  show_phase(sys1, [1-step, step], 1.)

  plt.axvline(0, color='black', lw=1, zorder=-1)
  plt.axhline(0, color='black', lw=1, zorder=-1)
  plt.axvline(1, color='grey', ls='--', lw=1, zorder=-1)
  plt.xlim(-0.5, 1.5)
  plt.ylim(-2., 2.)
  plt.yticks([-1, 0, 1], fontsize=13)
  plt.xticks([0., 0.5, 1], fontsize=13)
  plt.tick_params(direction='in')

def demo_degen_equil():
  add_annotation(R'$\theta \, \ddot{\theta}-\dot{\theta}^{2}+\cos^{2}\frac{\pi \theta}{2}=0$', [0.38, 0.90])
  # add_annotation(R'(d) Bogdanov-Takens equilibrium', [0.05, -0.2], 14)
  add_annotation('(D)', [0.03, 0.05], 15)

  step = 1e-3
  sys1 = lambda x, dx: (dx**2 - np.cos(np.pi*x/2)**2) / x

  # left to sing
  for x0 in np.linspace(-0.1, -0.5, 5):
    show_phase(sys1, [x0, -step/2], 1e-3)
    show_phase(sys1, [x0, -step/2], 2.)

  for dx0 in np.arange(0.2, 2., 0.4):
    show_phase(sys1, [-0.6, -step/2], dx0)

  # between sing and degen Bogdanov-Takens
  for x0 in np.linspace(1 - 0.1, 0.1, 7):
    show_phase(sys1, [x0, step/2], 1e-3)
    show_phase(sys1, [x0, step/2], 2.0)

  # right to degen Bogdanov-Takens
  for x0 in np.linspace(1.1, 1.65, 4):
    show_phase(sys1, [x0, step], 1e-3)
    show_phase(sys1, [x0, step], 2.)

  for dx0 in np.linspace(0.4, 1.8, 6):
    show_phase(sys1, [1.65, step], dx0)

  plt.plot(1, 0, 'o', markersize=2, color='black')
  plt.axvline(0, color='black', lw=1, zorder=-1)
  plt.axhline(0, color='black', lw=1, zorder=-1)
  plt.axvline(1, color='grey', ls='--', lw=1, zorder=-1)
  plt.xlim(-0.5, 1.5)
  plt.ylim(-2., 2.)
  plt.yticks([-1, 0, 1], fontsize=13)
  plt.xticks([0., 0.5, 1], fontsize=13)
  plt.tick_params(direction='in')

def demo_non_perpmeable1():
  add_annotation(R'$\frac{\sin\pi \theta}{\pi} \, \ddot{\theta}-\cos(\pi \theta) \, \dot{\theta}^{2}+1=0$', [0.23, 0.90])
  # add_annotation(R'(e) non-permeable singularity', [0.05, -0.2], 14)
  add_annotation('(E)', [0.03, 0.05], 15)

  step = 1e-3
  sys1 = lambda x, dx: (np.cos(np.pi*x) * dx**2 - 1) * np.pi / np.sin(np.pi*x)

  # left to sing1
  for x0 in np.linspace(-0.5, -0.1, 6):
    show_phase(sys1, [x0, -step/2], 1e-3)

  for x0 in np.linspace(-0.2, -0.1, 2):
    show_phase(sys1, [x0, -step/2], 2.0)

  for dx0 in np.linspace(0.6, 2., 6):
    show_phase(sys1, [-0.5, -step/2], dx0)

  # between sings
  for x0 in np.linspace(0.1, 1 - 0.05, 12):
    show_phase(sys1, [x0, step/2], 1e-3)

  # right to sing2
  for x0 in np.linspace(1.05, 1.5, 8):
    show_phase(sys1, [x0, 1.5], 1e-3)

  plt.axvline(0, color='black', lw=1, zorder=-1)
  plt.axvline(1, color='grey', ls='--', lw=1, zorder=-1)
  plt.axhline(0, color='black', lw=1, zorder=-1)
  plt.xlim(-0.5, 1.5)
  plt.ylim(-2., 2.)
  plt.yticks([-1, 0, 1], fontsize=13)
  plt.xticks([0., 0.5, 1], fontsize=13)
  plt.tick_params(direction='in')

def demo_non_perpmeable2():
  add_annotation(R'$\frac{\sin\pi \theta}{\pi} \, \ddot{\theta}-\dot{\theta}^{2}+\cos\frac{\pi \theta}{2}=0$', [0.33, 0.90])
  # add_annotation(R'(f) non-permeable singularity', [0.05, -0.2], 14)
  add_annotation('(F)', [0.03, 0.05], 15)

  step = 1e-3
  sys1 = lambda x, dx: (dx**2 - np.cos(np.pi*x / 2)) * np.pi / np.sin(np.pi*x)

  # left to sing1
  for x0 in np.linspace(-0.5, -0.1, 6):
    show_phase(sys1, [x0, -step/2], 1e-3)
    show_phase(sys1, [x0, -step/2], 2.)

  for dx0 in np.linspace(0.5, 1.7, 6):
    show_phase(sys1, [-0.5, -step/2], dx0)

  # between sings
  for x0 in np.linspace(0.1, 1 - 0.05, 8):
    show_phase(sys1, [x0, step/2], 1e-3)
    show_phase(sys1, [x0, step/2], 2.)

  for x0 in np.linspace(1.5, 1 + 0.1, 5):
    show_phase(sys1, [x0, 1 + step/2], 1e-3)

  for dx0 in np.linspace(0.7, 2.0, 5):
    show_phase(sys1, [1.5, 1 + step/2], dx0)

  plt.axvline(0, color='black', lw=1, zorder=-1)
  plt.axvline(1, color='grey', ls='--', lw=1, zorder=-1)
  plt.axhline(0, color='black', lw=1, zorder=-1)
  plt.xlim(-0.5, 1.5)
  plt.ylim(-2., 2.)
  plt.yticks([-1, 0, 1], fontsize=13)
  plt.xticks([0., 0.5, 1], fontsize=13)
  plt.tick_params(direction='in')

def demo_non_perpmeable3():
  add_annotation(R'$\frac{\sin\pi \theta}{\pi} \, \ddot{\theta}- \cos{(\pi\theta)} \dot{\theta}^{2}+\cos\frac{\pi \theta}{2}=0$', [0.15, 0.90])
  # add_annotation(R'(f) non-permeable singularity', [0.05, -0.2], 14)
  add_annotation('(F)', [0.03, 0.05], 15)

  step = 1e-3
  sys1 = lambda x, dx: (np.cos(np.pi*x) * dx**2 - np.cos(np.pi*x / 2)) * np.pi / np.sin(np.pi*x)

  # left to sing1
  for x0 in np.linspace(-0.5, -0.1, 6):
    show_phase(sys1, [x0, -step/2], 1e-3)
    show_phase(sys1, [x0, -step/2], 2.)

  for dx0 in np.linspace(0.5, 1.7, 6):
    show_phase(sys1, [-0.5, -step/2], dx0)

  # between sings
  for x0 in np.linspace(0.1, 0.8, 8):
    show_phase(sys1, [x0, step/2], 1e-3)

  for x0 in np.linspace(0.84, 0.995, 6):
    show_phase(sys1, [x0, step/2], 1e-3)

  for x0 in np.linspace(1.5, 1 + 0.1, 5):
    show_phase(sys1, [x0, 1 + step/2], 1e-3)
    show_phase(sys1, [x0, 1 + step/2], 2)

  for dx0 in np.linspace(0.7, 2.0, 5):
    show_phase(sys1, [1.5, 1 + step/2], dx0)

  plt.axvline(0, color='black', lw=1, zorder=-1)
  plt.axvline(1, color='grey', ls='--', lw=1, zorder=-1)
  plt.axhline(0, color='black', lw=1, zorder=-1)
  plt.xlim(-0.5, 1.5)
  plt.ylim(-2., 2.)
  plt.yticks([-1, 0, 1], fontsize=13)
  plt.xticks([0., 0.5, 1], fontsize=13)
  plt.tick_params(direction='in')

def main():
  fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True, sharex=True)
  plt.sca(axes[0,0])
  demo_isolated_sing()

  plt.sca(axes[0,1])
  demo_saddle_equil()

  plt.sca(axes[0,2])
  demo_saddle_sing()

  plt.sca(axes[1,0])
  demo_degen_equil()

  plt.sca(axes[1,1])
  demo_non_perpmeable1()

  plt.sca(axes[1,2])
  demo_non_perpmeable2()

  plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=-2.5)
  plt.savefig('fig/critical-points.pdf', dpi=150)
  plt.show()

def non_permeable_singularities_of_different_types():
  # plt.title('non-permeable singularity is between two saddles')

  step = 1e-3
  sys1 = lambda x, dx: (-dx**2 - np.cos(np.pi*x / 2)) / x

  # left to sing1
  # for x0 in np.linspace(-1.5, -1.1, 6):
  #   show_phase(sys1, [x0, -1 - step], 0.6)
  #   show_phase(sys1, [x0, -1 - step], 2.)

  for dx0 in np.linspace(1e-3, 2, 12):
    show_phase(sys1, [-1.5, -step], dx0)
  
  show_phase(sys1, [-1 - step, -1.5], 1e-3)

  for x0 in np.linspace(-1 + step, -0.1, 6):
    show_phase(sys1, [x0, -step], 1e-3)

  for x0 in np.linspace(0.1, 1 - step, 6):
    show_phase(sys1, [x0, step], 1e-3)

  for dx0 in np.linspace(1e-3, 2, 12):
    show_phase(sys1, [1.5, step], dx0)
  
  show_phase(sys1, [1 + step, 1.5], 1e-3)

  plt.axvline(0, color='black', lw=1, zorder=-1)
  plt.axhline(0, color='black', lw=1, zorder=-1)
  plt.xlim(-1.5, 1.5)
  plt.ylim(-2, 2)
  plt.show()

def draw_arrow(xy_tip, xy_tail, arc):
  arrowprops = {
    'arrowstyle': "Simple, tail_width=0.05, head_width=0.5, head_length=0.7",
    'connectionstyle': f"arc3,rad={arc}",
    'relpos': (1., 0.),
    'lw': 1.,
    'color': '#202020',
  }
  plt.annotate(
    '',
    xy = xy_tip, xycoords = 'data',
    xytext = xy_tail, textcoords='data',
    arrowprops = arrowprops
  )

def add_text(text, xy_data, fontsz=14):
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0.5,
    'alpha': 1
  }
  plt.annotate(
    text,
    xy = xy_data, xycoords = 'data',
    xytext = xy_data, textcoords='data',
    bbox = bbox,
    font = {'size': fontsz},
  )

def permeable_singularities():
  plt.figure('permeable singularities', figsize=(6, 4))

  def sys(x, dx):
    alpha = x * (x + 1)
    beta = -1
    gamma = 1 - x
    return (-beta * dx**2 - gamma) / alpha

  eps = 1e-3

  # left to saddle sing
  for x0 in np.linspace(-1.15, -2., 8):
    show_phase(sys, [x0, -2.], eps)
    show_phase(sys, [x0, -2.], 2.)

  show_phase(sys, [-1.0 - eps, -2], np.sqrt(2))

  # between singularities
  for x0 in np.linspace(-0.9, -0.1, 6):
    show_phase(sys, [x0, -eps], eps)
    show_phase(sys, [x0, -eps], 2.)
  
  show_phase(sys, [-1.0 + eps, -eps], np.sqrt(2))

  # between node sing and saddle equil
  for x0 in np.linspace(0.1, 0.8, 6):
    show_phase(sys, [x0, eps], eps)

  show_phase(sys, [1 - eps, eps], eps)

  # right to saddle equil
  for x0 in np.linspace(1.2, 2.0, 6):
    show_phase(sys, [x0, 2.0], eps)

  for dx0 in np.linspace(0.7, 2.0, 6):
    show_phase(sys, [2.0, eps], dx0)

  for x0 in np.linspace(0.1, 0.9, 5):
    show_phase(sys, [x0, eps], 2.0)

  for x0 in np.linspace(1.2, 1.6, 2):
    show_phase(sys, [x0, eps], 2.0)

  show_phase(sys, [1 + eps, 2.], eps)

  eps = 0.03

  # transition points
  draw_arrow((0.0 + eps, 1.0 + eps), (0.5, 1.6), -0.2)
  draw_arrow((-1 + eps, np.sqrt(2) + eps), (0.3, 1.6), 0.05)
  add_text('transtion\n points', (0.3, 1.5))

  draw_arrow((-1 - eps, 0.30), (-1.7, 0.30), 0)
  draw_arrow((-0 - eps, 0.15), (-1.7, 0.15), 0)
  add_text('forbidden\nregions', (-1.7, 0.15))

  draw_arrow((1 + eps, eps), (1.3, 0.5), -0.)
  add_text('saddle\nequil.', (1.2, 0.5))

  eps = 0.07
  forbidded_props = {
    'lw': 3,
    'color': '#B06060'
  }
  plt.plot([-1, -1], [-np.sqrt(2) + eps, np.sqrt(2) - eps], **forbidded_props)
  plt.plot([-1, -1], [np.sqrt(2) + eps, 2], **forbidded_props)

  plt.plot([0, 0], [-1 + eps, 1 - eps], **forbidded_props)
  plt.plot([0, 0], [1 + eps, 2], **forbidded_props)

  plt.axvline(1, color='grey', ls='--', lw=1, zorder=-1)
  plt.axhline(0, color='black', ls='-', lw=1, zorder=-1)
  plt.xlim(-1.8, 1.8)
  plt.ylim(-0.3, 2.)
  plt.yticks([])
  plt.xticks([-1, 0], ['saddle\nsingularity', 'node\nsingularity'], fontsize=14)
  plt.tick_params(direction='out')

  plt.tight_layout()
  plt.show()

def permeable_singularities_2():
  fig, axes = plt.subplots(1, 2, num='permeable singularities', figsize = (6, 4), sharex=True, sharey=True)

  eps = 1e-3

  # node singularity
  plt.sca(axes[0])

  def sys(x, dx):
    alpha = x
    beta = -1
    gamma = 1
    return (-beta * dx**2 - gamma) / alpha

  for x0 in np.linspace(-1, -0.1, 6):
    show_phase(sys, [x0, -eps], eps)
    show_phase(sys, [x0, -eps], 2.)

  for x0 in np.linspace(1, 0.1, 6):
    show_phase(sys, [x0, eps], eps)
    show_phase(sys, [x0, eps], 2.)

  for dx0 in np.linspace(0.5, 1.8, 6):
    show_phase(sys, [-1, -eps], dx0)
    show_phase(sys, [1, eps], dx0)

  eps = 0.07
  forbidded_props = {
    'lw': 3,
    'color': '#B06060'
  }
  plt.plot([0, 0], [-1 + eps, 1 - eps], **forbidded_props)
  plt.plot([0, 0], [1 + eps, 2], **forbidded_props)
  plt.plot([0, 0], [-1 - eps, -2], **forbidded_props)

  plt.axhline(0, lw=1, color='black', zorder=-2)

  plt.xlim(-1, 1)
  plt.ylim(-2, 2)

  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0.,
    'alpha': 1
  }
  plt.annotate(
    '(a) Node singularity.',
    (0, 0),
    (0.05, -0.08),
    textcoords='axes fraction',
    xycoords='axes fraction',
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )

  # saddle singularity
  plt.sca(axes[1])

  def sys(x, dx):
    alpha = x
    beta = 1
    gamma = -1
    return (-beta * dx**2 - gamma) / alpha

  eps = 1e-3

  for x0 in np.linspace(-1, -0.1, 6):
    show_phase(sys, [x0, -2], eps)
    show_phase(sys, [x0, -2], 2)

  for x0 in np.linspace(1, 0.1, 6):
    show_phase(sys, [x0, 2], eps)
    show_phase(sys, [x0, 2], 2)

  show_phase(sys, [-1, -eps], 1)
  show_phase(sys, [eps, 1], 1)
  
  eps = 0.07
  plt.plot([0, 0], [-1 + eps, 1 - eps], **forbidded_props)
  plt.plot([0, 0], [1 + eps, 2], **forbidded_props)
  plt.plot([0, 0], [-1 - eps, -2], **forbidded_props)

  plt.axhline(0, lw=1, color='black', zorder=-2)

  plt.xticks([])
  plt.yticks([])

  plt.xlim(-1, 1)
  plt.ylim(-2, 2)

  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0.,
    'alpha': 1
  }
  plt.annotate(
    '(b) Saddle singularity.',
    (0, 0),
    (0.05, -0.08),
    textcoords='axes fraction',
    xycoords='axes fraction',
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )

  # transition points
  arrowprops = {
    'arrowstyle': "Simple, tail_width=0.05, head_width=0.5, head_length=0.7",
    'connectionstyle': "arc3,rad=0",
    'relpos': (1., 0.),
    'lw': 1.,
    'color': '#202020',
  }
  plt.annotate(
    '',
    xy = (0.05, 1.05), xycoords = axes[0].transData,
    xytext = (0.9, 1.8), textcoords = axes[0].transData,
    arrowprops = arrowprops
  )
  plt.annotate(
    '',
    xy = (-0.05, 1.05), xycoords = axes[1].transData,
    xytext = (0.9, 1.8), textcoords = axes[0].transData,
    arrowprops = arrowprops
  )
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0.5,
    'alpha': 1
  }
  plt.annotate(
    'transition\npoints',
    (0, 0),
    (0.7, 1.4),
    textcoords=axes[0].transData,
    xycoords=axes[0].transData,
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )

  # forbidden region
  arrowprops = {
    'arrowstyle': "Simple, tail_width=0.05, head_width=0.5, head_length=0.7",
    'connectionstyle': "arc3,rad=0",
    'relpos': (1., 0.),
    'lw': 1.,
    'color': '#202020',
  }
  plt.annotate(
    '',
    xy = (0.03, -1.55), xycoords = axes[0].transData,
    xytext = (1., -1.55), textcoords = axes[0].transData,
    arrowprops = arrowprops
  )
  plt.annotate(
    '',
    xy = (-0.03, -1.55), xycoords = axes[1].transData,
    xytext = (1., -1.55), textcoords = axes[0].transData,
    arrowprops = arrowprops
  )
  bbox = {
    'boxstyle': 'round',
    'fc': '1.0',
    'lw': 0.5,
    'alpha': 1
  }
  plt.annotate(
    'forbidden\nregions',
    (0, 0),
    (0.7, -1.7),
    xycoords=axes[0].transData,
    textcoords=axes[0].transData,
    annotation_clip=False,
    bbox = bbox,
    font = {'size': 14},
  )

  plt.tight_layout(pad=1, h_pad=1, w_pad=-9.5)
  plt.savefig('fig/permeable-singularities.pdf', dpi=150)
  plt.show()

def solve_singular(alpha, beta, gamma, x0, dx0):
  def sys(s, st):
    x, y = st
    dx = alpha(x) * y
    dy = -beta(x) * y**2 - gamma(x)
    return np.array([dx, dy])

  sol = solve_ivp(sys, [100, 0], [x0, dx0], max_step=0.1)
  x = sol.y[0]
  dx = sol.y[1]
  ddx = np.zeros(x.shape)
  for i in range(x.shape[0]):
    ddx[i] = (-beta(x[i]) * dx[i]**2 - gamma(x[i])) / alpha(x[i])

  t = np.zeros(x.shape)
  t[1:] = np.diff(x) * 2 / (dx[1:] + dx[:-1])
  t = np.cumsum(t)
  return t, x, dx, ddx

def non_permeable_singularity():
  eps = 1e-3

  alpha = lambda x: x
  beta = lambda x: -2
  gamma = lambda x: -1*x

  def sys(x, dx):
    return (-beta(x) * dx**2 - gamma(x)) / alpha(x)

  for x0 in np.linspace(-1, -0.1, 6):
    show_phase(sys, [x0, -eps], eps)
    show_phase(sys, [x0, -eps], 2.)

  for x0 in np.linspace(1, 0.1, 6):
    show_phase(sys, [x0, eps], eps)
    show_phase(sys, [x0, eps], 2.)

  for dx0 in np.linspace(0.5, 2., 3):
    show_phase(sys, [-1., -eps], dx0)
    show_phase(sys, [1., eps], dx0)

  plt.axhline(0, color='grey', ls='--', lw=1)
  plt.axvline(0, color='grey', ls='--', lw=1)
  plt.xlim(-1, 1)
  plt.ylim(-2, 2)

  t, x, dx, ddx = solve_singular(alpha, beta, gamma, -1, 1)
  plt.figure()
  plt.plot(t, x)
  plt.plot(t, dx)
  plt.plot(t, ddx)
  plt.grid(True, ls='--', color='grey', zorder=-1)
  plt.show()

if __name__ == '__main__':
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  main()
  # permeable_singularities_2()
  # non_permeable_singularity()
