import matplotlib.pyplot as plt
from matplotlib import animation, rc
from furuta_pendulum.dynamics.parameters import FurutaPendulumPar, furuta_pendulum_param_default
import numpy as np
from common.rotations import rotmat, rot_z
from common.trajectory import Trajectory
from scipy.interpolate import make_interp_spline


class FurutaPendulumVis:
  def __init__(self, par : FurutaPendulumPar, color='blue'):
    self.get_vert_pos = vertex_pos_functor(par)
    self.__create_axis(par)
    self.__create_links([0, 0], color)

  def __create_axis(self, par : FurutaPendulumPar):
    ez = np.array([0, 0, 0.05])
    l = par.link_1_orient @ ez
    p = par.joint_1_pos
    pts = np.array([p - l, p + l])
    self.axis, = plt.plot(pts[:,0], pts[:,1], pts[:,2], ls='--', lw=2, color='black', alpha=0.5)

  def __create_links(self, q : np.ndarray, color):
    pts = self.get_vert_pos(q)
    links, = plt.plot(pts[:,0], pts[:,1], pts[:,2], '-', color=color, lw=6, alpha=1)
    joints, = plt.plot(pts[:,0], pts[:,1], pts[:,2], 'o', color='black', markersize=12, alpha=1)
    self.links = links
    self.joints = joints
    
  def elems(self):
    return self.links, self.joints, self.axis

  def set_links_properties(self, **kwargs):
    self.links.set(**kwargs)

  def set_joints_properties(self, **kwargs):
    self.joints.set(**kwargs)

  def move(self, q : np.ndarray):
    pts = self.get_vert_pos(q)
    self.links.set_data_3d(pts[:,0], pts[:,1], pts[:,2])
    self.joints.set_data_3d(pts[:,0], pts[:,1], pts[:,2])

def vertex_pos_functor(par : FurutaPendulumPar):
  ex = np.array([1., 0., 0.])
  joint_1_pos = par.joint_1_pos
  joint_2_pos = par.joint_2_pos
  link_2_length = par.link_2_length
  link_1_orient = par.link_1_orient
  link_2_orient = par.link_2_orient

  def compute_vertex_pos(q):
    q1, q2 = q
    R1 = link_1_orient @ rot_z(q1)
    R2 = R1 @ link_2_orient @ rot_z(q2)
    p1 = joint_1_pos
    p2 = joint_1_pos + R1 @ joint_2_pos
    p3 = joint_1_pos + R1 @ joint_2_pos + R2 @ ex * link_2_length
    return np.array([p1, p2, p3])
  
  return compute_vertex_pos

def init_axes(center : np.array, areasz : float, fig=None):
  if fig is None:
    fig = plt.figure()
  ax = fig.add_subplot(projection='3d', proj_type='persp')
  cx,cy,cz = center
  ax.autoscale(enable=False)
  ax.set_xbound(cx - areasz/2, cx + areasz/2)
  ax.set_ybound(cy - areasz/2, cy + areasz/2)
  ax.set_zbound(cz - areasz/2, cz + areasz/2)
  return ax

def get_configurations_occupancy_box(configurations : np.ndarray, par : FurutaPendulumPar):
  vert_pos = vertex_pos_functor(par)

  def get_bounds(q):
    pts = vert_pos(q)
    pmin = np.min(pts, axis=0)
    pmax = np.max(pts, axis=0)
    return pmin, pmax

  pmax, pmin = get_bounds(configurations[0])

  for q in configurations[1:]:
    bounds = get_bounds(q)
    pmin = np.minimum(bounds[0], pmin)
    pmax = np.maximum(bounds[1], pmax)
  
  return pmin, pmax

def get_maximum_occupancy_box(par : FurutaPendulumPar):
  configurations = 2 * np.pi * np.random.rand(100, 2)
  return get_configurations_occupancy_box(configurations, par)

def get_view_area(par : FurutaPendulumPar, configurations=None):
  if configurations is None:
    pmin, pmax = get_maximum_occupancy_box(par)
  else:
    pmin, pmax = get_configurations_occupancy_box(configurations, par)

  pc = (pmin + pmax) / 2
  w = np.max(pmax - pmin)
  return pc, w

def draw(q, par : FurutaPendulumPar, color='blue'):
  fig = plt.figure(figsize=(6, 6))
  center, areasz = get_view_area(par)
  ax = init_axes(center, areasz, fig)
  pend = FurutaPendulumVis(par, color=color)
  pend.move(q)

def motion_schematic(traj : Trajectory, par : FurutaPendulumPar):
  d = traj.phase - traj.phase[0]
  d = np.linalg.norm(d, axis=1)
  i, = np.nonzero(d < 1e-5)
  i = i[1]
  q1 = traj.coords[0]
  q2 = traj.coords[i//4]
  q3 = traj.coords[i//2]

  fig = plt.figure('Furuta Pendulum Schematic', figsize=(6, 5))
  plt.grid(True)
  center, areasz = get_view_area(par, traj.coords)
  ax = init_axes(center, areasz, fig)
  vis1 = FurutaPendulumVis(par, color='#3030E0')
  vis1.move(q1)
  vis2 = FurutaPendulumVis(par, color='#3030C0')
  vis2.move(q2)
  vis3 = FurutaPendulumVis(par, color='#3030A0')
  vis3.move(q3)

  plt.tight_layout()
  return fig

def animate(traj : Trajectory, par : FurutaPendulumPar, fps=60, speedup=1, videopath=None):
  q = traj.coords
  t = traj.time
  qfun = make_interp_spline(t, q, k=1)
  center, areasz = get_view_area(par, q)

  fig = plt.figure('Furuta Pendulum Sim', figsize=(8, 7))
  ax = init_axes(center, areasz, fig)
  vis = FurutaPendulumVis(par)

  animtime = (t[-1] - t[0]) / speedup
  nframes = int(animtime * fps)

  def drawframe(iframe):
    ti = speedup * iframe / fps + t[0]
    vis.move(qfun(ti))
    return vis.elems()

  anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=1000/fps, blit=True)
  plt.tight_layout()

  rc('animation', html='jshtml')
  if videopath:
    anim.save(videopath, fps=fps, bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

  return anim
