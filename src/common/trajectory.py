from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class Trajectory:
  time : np.ndarray
  phase : np.ndarray
  control : Optional[np.ndarray]

  def __post_init__(self):
    self.phase = np.array(self.phase, float)
    self.time = np.array(self.time, float)
    nt, nd = self.phase.shape
    assert nd % 2 == 0
    assert self.time.shape == (nt,)

    if self.control is not None:
      self.control = np.array(self.control, float)
      assert self.control.shape[0] == nt

  @property
  def dim(self):
    return self.phase.shape[1]
  
  def __getitem__(self, idx):
    if isinstance(idx, int):
      idx = slice(idx, idx + 1, None)

    if isinstance(idx, slice):
      return Trajectory(
        time = self.time.__getitem__(idx),
        phase = self.phase.__getitem__(idx),
        control = None if self.control is None else self.control.__getitem__(idx)
      )

    assert False

  @property
  def coords(self):
    ndim = self.dim
    return self.phase[:,0:ndim//2]
  
  @property
  def vels(self):
    ndim = self.dim
    return self.phase[:,ndim//2:]
  
  @property
  def duration(self):
    return self.time[-1] - self.time[0]
  
  def __len__(self):
    return len(self.time)
  
def compute_time(coords : np.ndarray, velocities : np.ndarray):
  dx = np.diff(coords, axis=0)
  mv = velocities[0:-1,:] + velocities[1:,:]
  dt = 2 * np.sum(dx * dx, axis=1) / np.sum(dx * mv, axis=1)
  t = np.zeros(len(dt) + 1)
  t[1:] = np.cumsum(dt)
  return t

def make_traj(coords : np.ndarray, velocities : np.ndarray, time=None, control=None):
  assert coords.shape == velocities.shape

  match coords.shape:
    case (n, d): pass
    case (n,):
      coords = np.reshape(coords, (n, 1))
      velocities = np.reshape(velocities, (n, 1))
    case _: assert False

  if time is None:
    time = compute_time(coords, velocities)

  phase = np.concatenate((coords, velocities), axis=1)
  return Trajectory(
    time = time,
    phase = phase,
    control = control
  )

def traj_reverse(tr : Trajectory) -> Trajectory:
  R"""
    Reverse trajectory in time
  """
  phase = np.zeros(tr.phase.shape)
  nq = tr.dim // 2
  phase[:,:nq] = tr.coords[::-1]
  phase[:,nq:] = -tr.vels[::-1]
  if tr.control is not None:
    control = tr.control[::-1]
  else:
    control = None
  time = tr.time[-1] - tr.time[::-1]
  return Trajectory(
    time = time,
    phase = phase,
    control = control
  )

def traj_join(tr1 : Trajectory, tr2 : Trajectory):
  R"""
    Contatenate two trajectories
  """
  controls_valid = None not in [tr1.control, tr2.control]
  phase2 = tr2.phase
  control2 = tr2.control

  if np.allclose(tr1.coords[-1], tr2.coords[0]):
    phase1 = tr1.phase[:-1]
    time1 = tr1.time[:-1]
    time2 = tr1.time[-1] - tr2.time[0] + tr2.time
    if controls_valid:
      control1 = tr1.control[:-1]
  else:
    phase1 = tr1.phase
    time1 = tr1.time
    control1 = tr1.control
    dx = tr2.coords[0] - tr1.coords[-1]
    mv = (tr2.vels[0,...] + tr1.vels[-1,...]) / 2
    dt = dx @ dx / (mv @ dx)
    time2 = tr1.time[-1] - tr2.time[0] + dt + tr2.time
    control1 = tr1.control

  time = np.concatenate((time1, time2))
  phase = np.concatenate((phase1, phase2))
  control = np.concatenate((control1, control2)) if controls_valid else None

  return Trajectory(
    time = time,
    phase = phase,
    control = control
  )

def traj_forth_and_back(tr : Trajectory) -> Trajectory:
  nt,nd = tr.phase.shape
  nx = nd // 2
  phase = np.zeros((2 * nt - 1, nd))
  phase[:nt-1] = tr.phase[:-1]
  phase[nt-1:,0:nx] = tr.coords[::-1]
  phase[nt-1:,nx:] = -tr.vels[::-1]

  time = np.zeros(2 * nt - 1)
  time[: nt - 1] = tr.time[:-1]
  time[nt - 1 :] = 2 * tr.time[-1] - tr.time[::-1]

  if tr.control is not None:
    _,nu = tr.control.shape
    control = np.zeros(2 * nt - 1, nu)
    control[0:nt-1] = tr.control[:-1]
    control[nt-1:] = tr.control[::-1]
  else:
    control = None
  
  return Trajectory(
    time = time,
    phase = phase,
    control = control
  )

def traj_repeat(tr : Trajectory, ntimes=2) -> Trajectory:
  assert np.allclose(tr.coords[0], tr.coords[-1])
  n,d = tr.phase.shape
  phase = np.zeros(((n - 1) * ntimes + 1, d))
  time = np.zeros((n - 1) * ntimes + 1)

  if tr.control is not None:
    _,du = tr.control.shape
    control = np.zeros(((n - 1) * ntimes + 1, du))
  else:
    control = None

  for i in range(ntimes):
    i1 = (n - 1) * i
    i2 = i1 + n
    phase[i1:i2, 0 : d//2] = tr.coords
    phase[i1:i2, d//2 : ] = tr.vels
    time[i1:i2] = tr.time[-1] * i + tr.time

    if control is not None:
      control[i1:i2] = tr.control[0:-1]
  
  return Trajectory(
    time = time,
    phase = phase,
    control = control
  )
  
def test_traj1():
  t = np.linspace(0, 2, 11)
  x = np.array([np.sin(t), np.cos(t)]).T
  tr = Trajectory(t, x, None)
  tr1 = tr[1:5]
  assert np.allclose(tr1.coords, x[1:5,0:1])
  assert np.allclose(tr1.vels, x[1:5,1:2])
  assert np.allclose(tr1.time, t[1:5])
  assert tr1.control is None

def test_traj2():
  t = np.linspace(0, 2, 11)
  x = np.array([np.sin(t), np.cos(t)]).T
  u = t**2
  tr = Trajectory(t, x, u)
  tr1 = tr[1:]
  assert np.allclose(tr1.control, u[1:])

def test_traj_join1():
  t1 = np.linspace(0, 2, 11)
  x1 = np.array([np.sin(t1), np.cos(t1)]).T
  tr1 = Trajectory(t1, x1, None)
  t2 = np.linspace(0, 3, 15)
  x2 = np.array([np.sin(t2 + 2), np.cos(t2 + 2)]).T
  tr2 = Trajectory(t2, x2, None)
  tr = traj_join(tr1, tr2)

  assert np.all(np.diff(tr.time) > 0)
  assert np.allclose(tr.time[-1], t1[-1] + t2[-1])
  n1, = t1.shape
  assert np.allclose(tr[n1-1].phase, x1[-1,:])
  assert np.allclose(tr[n1].phase, x2[1,:])

def test_traj_join2():
  t1 = np.linspace(0, 2, 11)
  x1 = np.array([
    np.sin(t1),
    np.sin(2*t1),
    np.cos(t1),
    2*np.cos(2*t1)
  ]).T
  tr1 = Trajectory(t1, x1, None)
  t2 = np.linspace(0.0, 3, 15)
  dt = 0.1
  x2 = np.array([
    np.sin(t2 + t1[-1] + dt),
    np.sin(2*(t2 + t1[-1] + dt)),
    np.cos(t2 + t1[-1] + dt),
    2*np.cos(2*(t2 + t1[-1] + dt))
  ]).T
  tr2 = Trajectory(t2, x2, None)
  tr = traj_join(tr1, tr2)

  assert np.all(np.diff(tr.time) > 0)
  assert np.allclose(tr.time[-1], t1[-1] + t2[-1] + dt, atol=1e-3)
  n1, = t1.shape
  assert np.allclose(tr[n1 - 1].phase, x1[-1,:])
  assert np.allclose(tr[n1].phase, x2[0,:])

def test_traj_reverse():
  t1 = np.linspace(1, 4, 11)
  x1 = np.array([
    np.sin(t1),
    np.sin(2*t1),
    np.cos(t1),
    2*np.cos(2*t1)
  ]).T
  tr1 = Trajectory(t1, x1, None)
  tr2 = traj_reverse(tr1)
  assert tr2.duration == tr1.duration
  assert np.allclose(tr1.coords[::-1], tr2.coords)
  assert np.allclose(tr1.vels[::-1], -tr2.vels)

def test_traj_forth_and_back():
  t = np.linspace(np.pi/2, 3*np.pi/2)
  v = np.cos(t)
  x = np.sin(t)
  tr = make_traj(x, v, t)
  tr2 = traj_forth_and_back(tr)
  assert np.allclose(2 * tr.duration, tr2.duration)
  assert np.allclose(tr2.coords[-1], tr.coords[0])
  assert np.allclose(tr2.coords[-8], tr.coords[7])
  assert np.allclose(tr2.vels[-8], -tr.vels[7])

def get_test_traj(duration, npts):
  t = np.linspace(0, duration, npts)
  x = np.array([
    np.sin(t),
    np.sin(2*t),
  ]).T
  v = np.array([
    np.cos(t),
    2*np.cos(2*t),
  ]).T
  return make_traj(x, v, t)

def test_make_traj1():
  tr1 = get_test_traj(4., 175)
  tr2 = make_traj(tr1.coords, tr1.vels)
  assert np.allclose(tr1.time, tr2.time, atol=1e-3)

def test_traj_repeat():
  import matplotlib.pyplot as plt
  tr1 = get_test_traj(2*np.pi, 175)
  tr2 = traj_repeat(tr1, 3)
  assert np.allclose(3 * tr1.duration, tr2.duration)
  assert np.allclose(tr1.coords[0], tr2.coords[-1])
  dt = np.mean(np.diff(tr1.time))
  assert np.allclose(np.diff(tr2.time), dt)

if __name__ == '__main__':
  test_traj1()
  test_traj2()
  test_traj_join1()
  test_traj_join2()
  test_traj_reverse()
  test_traj_forth_and_back()
  test_make_traj1()
  test_traj_repeat()
