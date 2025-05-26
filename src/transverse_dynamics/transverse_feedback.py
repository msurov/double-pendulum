from transverse_dynamics.transverse_dynamics import TransverseDynamics
from common.lqr import lqr_ltv, lqr_ltv_periodic
from scipy.interpolate import make_interp_spline
from dataclasses import dataclass
import numpy as np


@dataclass
class LTVSystem:
  t : np.ndarray
  A : np.ndarray
  B : np.ndarray

def tabulate_linsys(dynamics : TransverseDynamics, npts = 100) -> LTVSystem:
  theta = np.linspace(dynamics.transverse_coords.theta_min, dynamics.transverse_coords.theta_max, npts)
  n, m = dynamics.B_expr.shape
  A = np.zeros((npts, n, n))
  B = np.zeros((npts, n, m))

  for i in range(npts):
    A[i,:,:] = dynamics.A_fun(theta[i])
    B[i,:,:] = dynamics.B_fun(theta[i])

  return LTVSystem(t = theta, A = A, B = B)

@dataclass
class TranverseFeedbackControllerPar:
  Q : np.ndarray
  R : np.ndarray
  nsteps : int
  S : np.ndarray = None

class TranverseFeedbackController:
  def __init__(self, dynamics : TransverseDynamics, par : TranverseFeedbackControllerPar, **integ_args):
    theta = np.linspace(
      dynamics.transverse_coords.theta_min,
      dynamics.transverse_coords.theta_max, 
      par.nsteps,
    )

    if dynamics.transverse_coords.periodic:
      res = lqr_ltv_periodic(theta, dynamics.A_fun, dynamics.B_fun, par.Q, par.R, **integ_args)
      if res is None:
        assert False, "Can't solve LQR"
      bc_type = 'periodic'
    else:
      assert par.S is not None
      res = lqr_ltv(theta, dynamics.A_fun, dynamics.B_fun, par.Q, par.R, par.S, **integ_args)
      if res is None:
        assert False, "Can't solve LQR"
      bc_type = None

    K, P = res
    self.Ksp = make_interp_spline(theta, K, k=3, bc_type=bc_type)
    self.Psp = make_interp_spline(theta, P, k=3, bc_type=bc_type)
    self.coords_transform = dynamics.transverse_coords.forward_transform_fun
    self.u_ref = dynamics.transverse_coords.usp
    self.state = None
  
  def compute_stab_control(self, theta : float, xi : np.ndarray) -> np.ndarray:
    K = self.Ksp(theta)
    u = K @ xi
    return u

  def compute(self, x : np.ndarray, full_output=False) -> np.ndarray:
    val = self.coords_transform(x)
    theta = float(val[0])
    xi = np.reshape(val[1:], (-1,))
    K = self.Ksp(theta)
    u_stab = K @ xi
    u = self.u_ref(theta) + u_stab
    P = self.Psp(theta)
    V = xi.T @ P @ xi / 2

    self.state = [theta, *xi, *u_stab, V]

    if full_output:
      return u, u_stab, theta, xi, V

    return u

  def __call__(self, _, x):
    return self.compute(x)
