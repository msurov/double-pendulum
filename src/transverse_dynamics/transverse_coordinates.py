"""
transverse_dynamics
===================

Classes
-------
TransverseCoordinates : implemets methods for coordinates transform from Cartesian n-dim space to
  cylindrical S1 x Rn-1. The transformation works for stabilizing periodic trajectories.
TransverseDynamics : implements methods for computing dynamics in the given transverse coordinates

TODO:
  * Adapt the class TransverseDynamics for use with abstract dynamical system DynamicalSystem
  * Imeplement several base classes with common interface for constructing transverse coordinates
"""

from common.mechsys import MechanicalSystem
from common.trajectory import Trajectory
from common.bspline_sym import (
  MXSpline,
  make_interp_spline
)
from common.numpy_utils import (
  cont_angle, 
  is_ascending
)
from typing import Tuple
import casadi as ca
import numpy as np
from dataclasses import dataclass


@dataclass
class TransverseCoordinatesPar:
  transverse_projection_mat : np.ndarray
  proj_plane_x : np.ndarray
  proj_plane_y : np.ndarray
  proj_plane_origin : np.ndarray

class TransverseCoordinates:
  R"""
    Forward transformation
    \[
      \theta&=\arctan\frac{b\left(x-x_{0}\right)}{a\left(x-x_{0}\right)}, \\
      \xi_{1..2n-2}&=T\cdot\left(x-x_{*}\left(\theta\right)\right), \\
      \xi_{2n-1}&=b\left(x-x_{0}\right)\sin\theta+a\left(x-x_{0}\right)\cos\theta
    \]
    The inverse transformation is the solution of the LAE wrt x
    \[
      \xi_{2n-1}+\left(\cos\theta a+\sin\theta b\right)^{T}x_{0}&=\left(\sin\theta b+\cos\theta a\right)^{T}x\\
      \left(\cos\theta b-\sin\theta a\right)^{T}x_{0}&=\left(\cos\theta b-\sin\theta a\right)^{T}x\\
      \xi_{1..2n-2}+Tx_{*}\left(\theta\right)&=Tx
    \]

    The angle theta is the curve paramer is computed by the curve projected onto the plane 
    spanned by the vectors [proj_plane_x, proj_plane_y]
  """

  def __init__(self, traj : Trajectory, par : TransverseCoordinatesPar):
    self.par = par

    n = traj.dim
    assert par.transverse_projection_mat.shape == (n - 2, n)
    assert par.proj_plane_x.shape == (n,)
    assert par.proj_plane_y.shape == (n,)
    assert par.proj_plane_origin.shape == (n,)

    self.x = ca.MX.sym('x', n)
    self.theta = ca.MX.sym('theta')
    self.xi = ca.MX.sym('xi', n-1)

    self.x_ref = None
    self.u_ref = None
    self.__init_curve(traj)

    self.inverse_transform_expr = None
    self.inverse_jac_expr = None
    self.__init_inverse_transform()
    self.inverse_transform_fun = ca.Function('inverse_transform', [self.theta, self.xi], [self.inverse_transform_expr])
    self.inverse_jac_fun = ca.Function('inverse_jac', [self.theta, self.xi], [self.inverse_jac_expr])

    self.xi_expr = None
    self.theta_expr = None
    self.forward_transform_expr = None
    self.forward_jac_expr = None
    self.__init_forward_transform()
    self.forward_transform_fun = ca.Function('forward_transform', [self.x], [self.forward_transform_expr])
    self.forward_jac_fun = ca.Function('forward_jac', [self.x], [self.forward_jac_expr])

  def __init_curve(self, traj : Trajectory):
    x = (traj.phase - self.par.proj_plane_origin) @ self.par.proj_plane_x
    y = (traj.phase - self.par.proj_plane_origin) @ self.par.proj_plane_y  
    theta = np.arctan2(-y, x)
    cont_angle(theta)
    self.theta_min = theta[0]
    self.theta_max = theta[-1]

    assert is_ascending(theta)
    if np.all(traj.phase[-1] == traj.phase[0]):
      bc_type = 'periodic'
      assert np.allclose(theta[-1] - theta[0], 2 * np.pi)
    else:
      bc_type = None

    self.xsp = make_interp_spline(theta, traj.phase, k=5, bc_type=bc_type)
    self.x_ref = MXSpline(self.xsp)
    self.x_ref_expr = self.x_ref(self.theta)

    self.usp = make_interp_spline(theta, traj.control, k=3, bc_type=bc_type)
    self.u_ref = MXSpline(self.usp)
    self.u_ref_expr = self.u_ref(self.theta)

  def __verify_solvability(self, A):
    npts = 1000
    det_vals = np.zeros(npts)
    theta = np.linspace(self.theta_min, self.theta_max, npts)

    for i in range(npts):
      A_val = ca.substitute(A, self.theta, theta[i])
      A_val = ca.substitute(A_val, self.xi, ca.DM.zeros(*self.xi.shape))
      A_val = ca.evalf(A_val)
      det_vals[i] = ca.det(A_val)

    det_sign = np.sign(det_vals[0])
    det_vals = det_sign * det_vals
    assert np.all(det_vals > 1e-5), "Transverse parameters are incorrect, try to change projection matrix"

  def __init_inverse_transform(self):
    R"""
      x = x(theta, xi)
    """
    a = ca.DM(self.par.proj_plane_x).T
    b = ca.DM(self.par.proj_plane_y).T
    x0 = ca.DM(self.par.proj_plane_origin)
    T = ca.DM(self.par.transverse_projection_mat)
    theta = self.theta
    xi = self.xi
    x_ref_expr = self.x_ref_expr

    A = ca.vertcat(
      a * ca.sin(theta) + b * ca.cos(theta),
      a * ca.cos(theta) - b * ca.sin(theta),
      T
    )
    B = ca.vertcat(
      (a * ca.sin(theta) + b * ca.cos(theta)) @ x0,
      (a * ca.cos(theta) - b * ca.sin(theta)) @ x_ref_expr + xi[-1],
      xi[0:-1] + T @ self.x_ref_expr
    )
    self.__verify_solvability(A)

    inverse_transform_expr = ca.solve(A, B)
    self.inverse_transform_expr = inverse_transform_expr

    inverse_jac_expr = ca.horzcat(
      ca.jacobian(inverse_transform_expr, theta),
      ca.jacobian(inverse_transform_expr, xi)
    )
    self.inverse_jac_expr = inverse_jac_expr

  def __init_forward_transform(self):
    R"""
      theta = theta(x)
      xi = xi(x)
    """
    a = ca.DM(self.par.proj_plane_x).T
    b = ca.DM(self.par.proj_plane_y).T
    x0 = ca.DM(self.par.proj_plane_origin)
    x = self.x
    T = ca.DM(self.par.transverse_projection_mat)

    theta = -ca.arctan2(b @ (x - x0), a @ (x - x0))
    x_ref = self.x_ref(theta)
    xi = ca.vertcat(
      T @ (x - x_ref),
      (a * ca.cos(theta) - b * ca.sin(theta)) @ (x - x_ref)
    )
    self.xi_expr = xi
    self.theta_expr = theta
    self.forward_transform_expr = ca.vertcat(self.theta_expr, self.xi_expr)
    self.forward_jac_expr = ca.jacobian(self.forward_transform_expr, x)

def compute_theta(phase : np.ndarray, par : TransverseCoordinatesPar) -> np.ndarray:
  x = (phase - par.proj_plane_origin) @ par.proj_plane_x
  y = (phase - par.proj_plane_origin) @ par.proj_plane_y
  theta = -np.arctan2(y, x)
  if isinstance(theta, np.ndarray):
    cont_angle(theta)
  return theta

def compute_transverse(phase : np.ndarray, coords : TransverseCoordinates) -> Tuple[np.ndarray, np.ndarray]:
  match np.shape(phase):
    case (npts, dim):
      arr = np.array([coords.forward_transform_fun(e) for e in phase], float)
      theta = arr[:,0,0]
      cont_angle(theta)
      xi = arr[:,1:,0]
    case (dim,):
      arr = coords.forward_transform_fun(phase)
      theta = float(arr[0])
      xi = np.reshape(arr[1:], (dim - 1,))
    case _: assert False

  return theta, xi

class TransverseDynamics:
  def __init__(self, dynamics : MechanicalSystem, coords : TransverseCoordinates):
    xdim,_ = dynamics.rhs_expr.shape
    udim,_ = dynamics.u.shape

    self.transverse_coords = coords

    self.u_stab = ca.MX.sym('u_stab', udim)
    self.xi = coords.xi
    self.theta = coords.theta

    self.D_xi_expr = None
    self.__init_transverse_dynamics(dynamics, coords)
    self.D_xi_fun = ca.Function('Dxi', [coords.theta, coords.xi, self.u_stab], [self.D_xi_expr])

    self.A_expr = None
    self.B_expr = None
    self.rest_expr = None
    self.__linearize_transverse_dynamics()
    self.A_fun = ca.Function('A', [coords.theta], [self.A_expr])
    self.B_fun = ca.Function('B', [coords.theta], [self.B_expr])
    self.rest_fun = ca.Function('B', [coords.theta], [self.rest_expr])

  def __init_transverse_dynamics(self, dynamics : MechanicalSystem, coords : TransverseCoordinates):
    dot_x = dynamics.rhs(
      coords.inverse_transform_expr,
      coords.u_ref_expr + self.u_stab
    )
    self.dot_x_expr = dot_x
    time_deriv_theta_xi = ca.solve(coords.inverse_jac_expr, dot_x)
    time_deriv_theta = time_deriv_theta_xi[0]
    time_deriv_xi = time_deriv_theta_xi[1:]
    self.time_deriv_theta_expr = time_deriv_theta
    self.time_deriv_xi_expr = time_deriv_xi
    D_xi_expr = time_deriv_xi / time_deriv_theta
    self.D_xi_expr = D_xi_expr

  def __linearize_transverse_dynamics(self):
    expr = ca.substitute(self.D_xi_expr, self.u_stab, ca.DM.zeros(*self.u_stab.shape))
    expr = ca.jacobian(expr, self.xi)
    expr = ca.substitute(expr, self.xi, ca.DM.zeros(*self.xi.shape))
    A = ca.simplify(expr)
    self.A_expr = A

    expr = ca.jacobian(self.D_xi_expr, self.u_stab)
    expr = ca.substitute(expr, self.u_stab, ca.DM.zeros(*self.u_stab.shape))
    expr = ca.substitute(expr, self.xi, ca.DM.zeros(*self.xi.shape))
    B = ca.simplify(expr)
    self.B_expr = B

    expr = ca.substitute(self.D_xi_expr, self.u_stab, ca.DM.zeros(*self.u_stab.shape))
    expr = ca.substitute(expr, self.xi, ca.DM.zeros(*self.xi.shape))
    rest = ca.simplify(expr)
    self.rest_expr = rest
  