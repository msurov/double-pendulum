import numpy as np
from bisect import bisect


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

def rotmat(a : float) -> np.ndarray:
  return np.array([
    [np.cos(a), -np.sin(a)],
    [np.sin(a), np.cos(a)]
  ])

def rotmat_deriv(a : float) -> np.ndarray:
  return np.array([
    [-np.sin(a), -np.cos(a)],
    [np.cos(a), -np.sin(a)]
  ])

def integrate_trapz(x : np.ndarray, y : np.ndarray) -> np.ndarray:
  s = np.zeros(y.shape)
  s[1:] = (y[1:] + y[:-1]) * np.diff(x) / 2
  s = np.cumsum(s)
  return s

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

def dot(a, b):
  return np.dot(a, b)

def skew(a, b):
  return a[0] * b[1] - a[1] * b[0]

def is_ascending(a : np.ndarray) -> bool:
  return np.all(np.diff(a) > 0)

def are_angles_close(a : np.ndarray, b : np.ndarray) -> bool:
  normed = 0.5 * (a - b) / np.pi
  rounded = np.round(normed)
  return np.allclose(normed, rounded)
