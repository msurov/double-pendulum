from .numpy_utils import normalized
import numpy as np


def wedge(v : np.ndarray) -> np.ndarray:
  x,y,z = v
  return np.array([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
  ])

def vee(A):
  return np.array([A[2,1], A[0,2], A[1,0]])

def rotmat(angle : float, axis : np.ndarray):
  I = np.eye(3)
  K = wedge(normalized(axis))
  R = I + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
  return R

def rot_x(angle):
  return rotmat(angle, np.array([1., 0., 0.]))

def rot_y(angle):
  return rotmat(angle, np.array([0., 1., 0.]))

def rot_z(angle):
  return rotmat(angle, np.array([0., 0., 1.]))
