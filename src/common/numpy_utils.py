import numpy as np
from bisect import bisect
from scipy.integrate import cumulative_simpson, cumulative_trapezoid
from typing import Union

def cont_angle(a : np.ndarray) -> np.ndarray:
  '''
    make the angle continuous
  '''
  for i in range(1, len(a)):
    da = a[i] - a[i-1]
    n = np.round(da / (2*np.pi))
    a[i] -= n * 2 * np.pi

def normalize_angle(a : float, angle_from=-np.pi, angle_to=np.pi) -> float:
  '''
    to the range [angle_from .. angle_to)
  '''
  n = np.floor((a - angle_from) / (angle_to - angle_from))
  return a - n * (angle_to - angle_from)

def skew(a : float, b : float) -> float:
  ax, ay = a
  bx, by = b
  return ax*by - ay*bx

def rotmat2d(a : float) -> np.ndarray:
  return np.array([
    [np.cos(a), -np.sin(a)],
    [np.sin(a), np.cos(a)]
  ])

def rotmat2d_deriv(a : float) -> np.ndarray:
  return np.array([
    [-np.sin(a), -np.cos(a)],
    [np.cos(a), -np.sin(a)]
  ])

def integrate_array(
    x : np.ndarray,
    y : np.ndarray,
    method : Union['simpson', 'trapz'] = 'simpson',
    I0 : Union[float, np.ndarray, None] = None
  ) -> np.ndarray:
  if method == 'simpson':
    return integrate_simp(x=x, y=y, I0=I0)
  elif method == 'trapz':
    return integrate_trapz(x=x, y=y, I0=I0)
  assert False, 'Unknown method ' + str(method)

def integrate_trapz(
    x : np.ndarray,
    y : np.ndarray,
    I0 : Union[float, np.ndarray, None] = None
  ) -> np.ndarray:
  npts, = x.shape
  assert y.shape[0] == npts
  res = cumulative_trapezoid(x=x, y=y, initial=0, axis=0)
  if I0 is not None:
    res += np.reshape(I0, (1,) + y.shape[1:])
  return res

def integrate_simp(
    x : np.ndarray,
    y : np.ndarray,
    I0 : Union[float, np.ndarray, None] = None
  ) -> np.ndarray:
  npts, = x.shape
  assert y.shape[0] == npts
  if I0 is None:
    I0 = 0 * y[0,...]
  I0 = np.reshape(I0, (1,) + y.shape[1:])
  return cumulative_simpson(x=x, y=y, initial=I0, axis=0)

def linear_interp(xx : np.ndarray, yy : np.ndarray, x : float) -> np.ndarray:
  i = bisect(xx, x)
  if i == 0:
      return yy[0,...]
  if i == len(xx):
      return yy[-1,...]

  y1 = yy[i - 1,...]
  y2 = yy[i,...]
  x1 = xx[i - 1]
  x2 = xx[i]
  y = y1 * (x2 - x) / (x2 - x1) + y2 * (x - x1) / (x2 - x1)
  return y

def normalized(v : np.ndarray) -> np.ndarray:
  return v / np.linalg.norm(v)

def dot(a : np.ndarray, b : np.ndarray) -> float:
  return np.dot(a, b)

def skew(a : np.ndarray, b : np.ndarray) -> float:
  return a[0] * b[1] - a[1] * b[0]

def wedge(a : np.ndarray, b : np.ndarray) -> np.ndarray:
  Q = np.outer(a, b)
  return Q - Q.T

def is_ascending(a : np.ndarray) -> bool:
  return np.all(np.diff(a) > 0)

def are_angles_close(a : np.ndarray, b : np.ndarray) -> bool:
  normed = 0.5 * (a - b) / np.pi
  rounded = np.round(normed)
  return np.allclose(normed, rounded)

def vectors_dif(a : np.ndarray, b : np.ndarray) -> float:
  a = np.reshape(a, (-1,))
  b = np.reshape(b, (-1,))
  return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def elementwise_dot(a : np.ndarray, b : np.ndarray) -> np.ndarray:
  return np.sum(a * b, axis=-1)
