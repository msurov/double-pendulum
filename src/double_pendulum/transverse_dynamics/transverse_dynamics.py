import casadi as ca
import numpy as np
from dataclasses import dataclass
from double_pendulum.dynamics import (
  DoublePendulumDynamics
)
from common.trajectory import (
  Trajectory, traj_affine_transform
)
from common.bspline_sym import MXSpline
from common.numpy_utils import (
  cont_angle, 
  is_ascending
)
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


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

    assert is_ascending(theta)
    if np.all(traj.phase[-1] == traj.phase[0]):
      bc_type = 'periodic'
      assert np.allclose(theta[-1] - theta[0], 2 * np.pi)
    else:
      bc_type = None

    self.__xsp = make_interp_spline(theta, traj.phase, k=5, bc_type=bc_type)
    self.x_ref = MXSpline(self.__xsp)
    self.x_ref_expr = self.x_ref(self.theta)

    self.__usp = make_interp_spline(theta, traj.control, k=3, bc_type=bc_type)
    self.u_ref = MXSpline(self.__usp)
    self.u_ref_expr = self.u_ref(self.theta)

  def __verify_solvability(self, A):
    A_val = ca.substitute(A, self.theta, 0)
    A_val = ca.substitute(A_val, self.xi, 0)
    A_val = ca.evalf(A_val)
    abs_det = abs(ca.det(A_val))
    assert abs_det > 1e-6, "Transverse parameters are incorrect, try to change projection matrix"

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
      ca.jacobian(inverse_transform_expr, xi),
      ca.jacobian(inverse_transform_expr, theta)
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

class TransverseDynamics:
  def __init__(self, dynamics : DoublePendulumDynamics, coords : TransverseCoordinates):
    xdim,_ = dynamics.rhs_expr.shape
    udim,_ = dynamics.u.shape

    self.u_stab = ca.MX.sym('u_stab', udim)
    self.xi = coords.xi
    self.theta = coords.theta

    self.D_xi_expr = None
    self.__init_transverse_dynamics()
    self.D_xi_fun = ca.Funcion('Dxi', [coords.theta, coords.xi, coords.w], [self.D_xi_expr])

    self.A_expr = None
    self.B_expr = None
    self.rest_expr = None
    self.__linearize_transverse_dynamics()

  def __init_transverse_dynamics(self, dynamics : DoublePendulumDynamics, coords : TransverseCoordinates):
    dot_x = dynamics.rhs(
      coords.inverse_transform_expr,
      coords.u_ref_expr + self.u_stab
    )
    dot_xi_theta = coords.inverse_jac_expr @ dot_x
    D_xi_expr = dot_xi_theta[0:n-1] / dot_xi_theta[n-1]
    self.D_xi_expr = D_xi_expr

  def __linearize_transverse_dynamics(self):
    expr = ca.substitute(self.D_xi_expr, self.u_stab, 0)
    expr = ca.jacobian(expr, self.xi)
    expr = ca.substitute(expr, self.xi, 0)
    A = ca.simplify(expr)
    self.A_expr = A

    expr = ca.jacobian(self.D_xi_expr, self.u_stab)
    expr = ca.substitute(expr, self.u_stab, 0)
    expr = ca.substitute(expr, self.xi, 0)
    B = ca.simplify(expr)
    self.B_expr = B

    expr = ca.substitute(self.D_xi_expr, self.u_stab, 0)
    expr = ca.substitute(expr, self.xi, 0)
    rest = ca.simplify(expr)
    self.rest_expr = rest
  