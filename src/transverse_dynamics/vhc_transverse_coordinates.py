from .transverse_dynamics import TransverseCoordinates
from common.bspline_sym import (
  MXSpline,
  make_interp_spline
)
from common.trajectory import Trajectory
import casadi as ca
import numpy as np
from common.numpy_utils import (
  cont_angle,
  is_ascending,
)

class VHCTransverseCoordinates(TransverseCoordinates):
  R"""
    VHC based transverse coordinates
  """
  def __init__(self, 
        traj : Trajectory, 
        vhc_implicit : ca.MX,      # y = h(q) = 0
        vhc_implicit_inv : ca.MX,  # q = hinv(y, x)
        free_var_idx : int,        # x = q[free_var_idx]
        free_var_origin : float    # x0
      ):

    self.periodic = True
    self.nq = vhc_implicit.sx_in()[0].size1()
    self.lam0 = free_var_origin
    self.lam_index = free_var_idx
    self.__init_traj(traj)
    self.__init_forward_transform(vhc_implicit)
    self.__init_inverse_transform(vhc_implicit_inv)

  def __init_forward_transform(self, vhc_implicit : ca.Function):
    """
      theta = theta(x)
      xi = xi(x)
    """
    lam0 = self.lam0
    nq = self.nq

    q = ca.MX.sym('q', nq)
    dq = ca.MX.sym('dq', nq)
    lam = q[self.lam_index]
    dlam = dq[self.lam_index]

    # 1. compute theta(x)
    theta_expr = ca.atan2(lam - lam0, dlam)

    # 2. compute rho(x)
    sin_theta = ca.sin(self.theta)
    cos_theta = ca.cos(self.theta)

    lam_ref = self.reduced_traj_expr[0]
    dlam_ref = self.reduced_traj_expr[1]

    rho_expr = (lam - lam_ref) * sin_theta + (dlam - dlam_ref) * cos_theta
    rho_expr = ca.substitute(rho_expr, self.theta, theta_expr)

    # 3. compute y(x)
    y_expr = vhc_implicit(q)

    # 4. compute dy(x)
    dy_expr = ca.jtimes(y_expr, q, dq)

    # 5. compose them into xi
    xi_expr = ca.vertcat(
      y_expr, dy_expr, rho_expr
    )

    # 6. make functions
    x = ca.vertcat(q, dq)
    theta_fun = ca.Function('theta', [x], [theta_expr])
    xi_fun = ca.Function('xi', [x], [xi_expr])

    self.theta_expr = theta_expr
    self.xi_expr = xi_expr
    self.forward_transform_expr = ca.vertcat(self.theta_expr, self.xi_expr)
    self.forward_jac_expr = ca.jacobian(self.forward_transform_expr, x)
    self.forward_transform_fun = ca.Function('transverse_forward_transform', [x], [self.forward_transform_expr])

  def __init_inverse_transform(self, vhc_implicit_inv : ca.Function):
    R"""
      x = x(theta, xi)
    """
    nq = self.nq
    self.xi = ca.MX.sym('xi', 2*nq - 1)
    y = self.xi[0:nq-1]
    dy = self.xi[nq-1:2*nq-2]
    rho = self.xi[2*nq-2]

    sin_theta = ca.sin(self.theta)
    cos_theta = ca.cos(self.theta)

    lam_ref = self.reduced_traj_expr[0]
    dlam_ref = self.reduced_traj_expr[1]

    # lambda = lambda(theta, xi)
    lam_expr = self.lam0 + rho * sin_theta + (lam_ref - self.lam0) * sin_theta**2 + dlam_ref * sin_theta * cos_theta
    # dlambda = dlambda(theta, xi)
    dlam_expr = rho * cos_theta + (lam_ref - self.lam0) * sin_theta * cos_theta + dlam_ref * cos_theta**2

    # q = q(y, lambda)
    y_tmp = ca.MX.sym('dummy-1', nq - 1)
    lam_tmp = ca.MX.sym('dummy-2')
    q_expr = vhc_implicit_inv(y_tmp, lam_tmp)
    dq_expr = ca.jtimes(q_expr, y_tmp, dy) + ca.jtimes(q_expr, lam_tmp, dlam_expr)

    q_expr = ca.substitute(q_expr, y_tmp, y)
    q_expr = ca.substitute(q_expr, lam_tmp, lam_expr)

    dq_expr = ca.substitute(dq_expr, y_tmp, y)
    dq_expr = ca.substitute(dq_expr, lam_tmp, lam_expr)

    self.inverse_transform_expr = ca.vertcat(q_expr, dq_expr)
    self.inverse_jac_expr = ca.horzcat(
      ca.jacobian(self.inverse_transform_expr, self.theta),
      ca.jacobian(self.inverse_transform_expr, self.xi)
    )

  def __init_traj(self, traj : Trajectory):
    nq = self.nq
    ilam = self.lam_index
    theta_arr = np.atan2(traj.coords[:,ilam] - self.lam0, traj.vels[:,ilam])
    theta_arr = np.reshape(theta_arr, (-1,))
    cont_angle(theta_arr)
    assert is_ascending(theta_arr)

    reduced_traj_sp = make_interp_spline(theta_arr, traj.phase[:,[ilam, nq + ilam]], bc_type='periodic')
    reduced_traj_mxsp = MXSpline(reduced_traj_sp)
    self.reduced_traj_sp = reduced_traj_sp
    self.reduced_traj_mxsp = reduced_traj_mxsp
    self.theta = ca.MX.sym('theta')
    self.reduced_traj_expr = reduced_traj_mxsp(self.theta)

    u_sp = make_interp_spline(theta_arr, traj.control, bc_type='periodic')
    u_mxsp = MXSpline(u_sp)
    self.usp = u_sp
    self.u_mxsp = u_mxsp
    self.u_ref_expr = u_mxsp(self.theta)

    self.theta_min = theta_arr[0]
    self.theta_max = theta_arr[-1]
