"""
transverse_dynamics
===================

Classes
-------
  TransverseCoordinates : implemets methods for coordinates transform from Cartesian n-dim space to a set of transverse coordinates
  TransverseDynamics : implements methods for computing dynamics in the given transverse coordinates
"""

from common.mechsys import MechanicalSystem
from common.numpy_utils import map_array
import casadi as ca
import numpy as np


class TransverseCoordinates:
  """
    Abstract transverse coordinates
  """
  def __init__(self):
    self.xi = None # MX-variable for transverse coordinates
    self.theta = None # MX-variable for traj projection theta
    self.x = None # MX-variable for phase coordinates

    self.theta_min = None # theta diap lower
    self.theta_max = None # theta diap upper

    self.inverse_transform_expr = None # x is expression of (theta,xi)
    self.inverse_transform_fun = None

    self.inverse_jac_expr = None
    self.inverse_jac_fun = None

    self.forward_transform_expr = None # (theta, xi) is expression of x
    self.forward_transform_fun = None

    self.forward_jac_expr = None
    self.forward_jac_fun = None

    self.usp = None # MX-spline for reference control
    self.u_ref_expr = None # u_ref_expr is expression of theta

def verify_transversality(transverse_coords : TransverseCoordinates):
  jac_expr = transverse_coords.inverse_jac_expr
  jac_expr = ca.substitute(jac_expr, transverse_coords.xi, 0)
  jac_fun = ca.Function('det_Jac', [transverse_coords.theta], [jac_expr])
  f = lambda w : np.linalg.det(jac_fun(w))
  theta = np.linspace(transverse_coords.theta_min, transverse_coords.theta_max)
  det_jac = map_array(f, theta)
  det_jac = det_jac * np.sign(det_jac[0])
  assert np.all(det_jac > 0)


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
