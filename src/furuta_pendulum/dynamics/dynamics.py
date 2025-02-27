from common.mechsys import MechanicalSystem
from common.casadi_utils import *
from .parameters import FurutaPendulumPar
import numpy as np


class FurutaPendulumDynamics(MechanicalSystem):
  def __init__(self, par : FurutaPendulumPar):
    self.q = ca.SX.sym('q', 2)
    self.dq = ca.SX.sym('dq', 2)
    self.u = ca.SX.sym('u')

    self.M_expr = self.get_kinetic_energy_mat(par)
    self.U_expr = self.get_potential_energy(par)
    self.G_expr = ca.jacobian(self.U_expr, self.q).T
    self.C_expr = self.compute_C(self.M_expr, self.q, self.dq)
    self.B_expr = ca.DM([[1], [0]])
    self.K_expr = self.dq.T @ self.M_expr @ self.dq / 2
    self.E_expr = self.K_expr + self.U_expr
    self.ddq_expr = ca.solve(self.M_expr, -self.C_expr @ self.dq - self.G_expr + self.B_expr * self.u)
    self.rhs_expr = ca.vertcat(self.dq, self.ddq_expr)

    self.M = ca.Function('M', [self.q], [self.M_expr])
    self.C = ca.Function('C', [self.q, self.dq], [self.C_expr])
    self.G = ca.Function('G', [self.q], [self.G_expr])
    self.B = ca.Function('B', [self.q], [self.B_expr])
    self.U = ca.Function('U', [self.q], [self.U_expr])
    self.K = ca.Function('K', [self.q, self.dq], [self.K_expr])
    self.E = ca.Function('E', [self.q, self.dq], [self.E_expr])
    self.rhs = ca.Function('rhs', [ca.vertcat(self.q, self.dq), self.u], [self.rhs_expr])

  def get_links_poses(self, par : FurutaPendulumPar):
    q1 = self.q[0]
    q2 = self.q[1]

    R1 = par.link_1_orient @ rot_z(q1)
    R2 = R1 @ par.link_2_orient @ rot_z(q2)
    p1 = par.joint_1_pos + R1 @ par.link_1_mass_center
    p2 = par.joint_1_pos + R1 @ par.joint_2_pos + R2 @ par.link_2_mass_center

    R1 = ca.simplify(R1)
    R2 = ca.simplify(R2)
    p1 = ca.simplify(p1)
    p2 = ca.simplify(p2)

    return p1, R1, p2, R2
  
  def get_vel_jac(self, par : FurutaPendulumPar):
    p1, R1, p2, R2 = self.get_links_poses(par)

    val1_jac = ca.jacobian(p1, self.q)
    val2_jac = ca.jacobian(p2, self.q)

    angvel1_jac = compute_angvel_jac(R1, self.q)
    angvel2_jac = compute_angvel_jac(R2, self.q)

    return val1_jac, val2_jac, angvel1_jac, angvel2_jac

  def get_kinetic_energy_mat(self, par : FurutaPendulumPar):
    val1_jac, val2_jac, angvel1_jac, angvel2_jac = self.get_vel_jac(par)
    M1 = val1_jac.T @ val1_jac * par.link_1_mass
    M2 = val2_jac.T @ val2_jac * par.link_2_mass
    M3 = angvel1_jac.T @ par.link_1_inertia_tensor @ angvel1_jac
    M4 = angvel2_jac.T @ par.link_2_inertia_tensor @ angvel2_jac
    M = ca.simplify(M1 + M2 + M3 + M4)
    return M

  def get_potential_energy(self, par : FurutaPendulumPar):
    g = par.gravity_accel
    p1, _, p2, _ = self.get_links_poses(par)
    U = p1[2] * g * par.link_1_mass + p2[2] * g * par.link_2_mass
    U = ca.simplify(U)
    return U

  def compute_C(self, M : ca.SX, q : ca.SX, dq : ca.SX) -> ca.SX:
    C1 = ca.jtimes(M, q, dq)
    C2 = 0.5 * ca.jacobian(M @ dq, q).T
    C = C1 - C2
    C = ca.simplify(C)
    return C

