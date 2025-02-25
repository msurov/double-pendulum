from double_pendulum.transverse_dynamics.transverse_dynamics import (
  TransverseDynamics,
  TransverseCoordinates,
  TransverseCoordinatesPar
)
from common.lqr import lqr_ltv
from dataclasses import dataclass
import numpy as np


@dataclass
class TranverseFeedbackControllerPar:
  Q : np.ndarray
  R : np.ndarray

class TranverseFeedbackController:
  def __init__(TranverseFeedbaskController):
    pass

def compute_linsys_mat(dynamics : TransverseDynamics, npts = 100):
  coords = dynamics.transverse_coords
  theta = np.linspace(coords.theta_min, coords.theta_max, npts)
  n, m = dynamics.B_expr.shape
  A = np.zeros((npts, n, n))
  B = np.zeros((npts, n, m))

  for i in range(npts):
    A[i,:,:] = dynamics.A_fun(theta[i])
    B[i,:,:] = dynamics.B_fun(theta[i])

  return A, B

def compute_transverse_lqr(dynamics : TransverseDynamics, par : TranverseFeedbackControllerPar):
  compute_linsys_mat()
