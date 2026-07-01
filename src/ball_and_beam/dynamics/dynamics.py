import casadi as ca
from ball_and_beam.dynamics.parameters import BallAndBeamPar
from typing import List
from common.mechsys import MechanicalSystem
from common.casadi_utils import rotmat2d


sin = ca.sin
cos = ca.cos
TP = ca.SX
jacobian = ca.jacobian
jtimes = ca.jtimes


def compute_C(M : TP, q : TP, dq : TP) -> TP:
  C1 = ca.jtimes(M, q, dq)
  C2 = 0.5 * ca.jacobian(M @ dq, q).T
  C = C1 - C2
  C = ca.simplify(C)
  return C

class BallAndBeamDynamics(MechanicalSystem):
  R"""
    Symbolic dynamics of cart pendulum system
    based on CasADi symbolic package
  """
  def __init__(self, par : BallAndBeamPar, auto_compute = False):
    self.par = par
  
    q = TP.sym('q', 2)
    dq = TP.sym('dq', 2)
    u = TP.sym('u', 1)

    self.q = q
    self.dq = dq
    self.u = u

    ey = ca.DM([[0], [1]])
    U = self.par.ball_mass * self.par.gravity_accel * ey.T @ self.get_ball_pos()

    if auto_compute:
      M = self.get_kinetic_energy_mat()
      M = ca.simplify(M)
      C = compute_C(M, q, dq)
      C = ca.simplify(C)
      G = ca.gradient(U, q)
      G = ca.simplify(G)
      K = self.get_kinetic_energy()
      K = ca.simplify(K)
      D = self.get_airdrag_force_mat()

    else:
      h = self.par.ball_center_displacement
      m_ball = self.par.ball_mass
      R_ball = self.par.ball_radius
      J_ball = self.par.ball_intertia
      J_beam = self.par.beam_inertia
      g = self.par.gravity_accel

      M = TP.zeros((2,2))
      M[0,0] = J_beam + J_ball + m_ball * (q[1]**2 + h**2)
      M[0,1] = \
      M[1,0] = -m_ball * h - J_ball / R_ball
      M[1,1] = m_ball + J_ball / R_ball**2

      C = TP.zeros((2,2))
      C[0,0] = 2 * m_ball * q[1] * dq[1]
      C[1,0] = -m_ball * q[1] * dq[0]

      G = TP.zeros((2,1))
      G[0,0] = m_ball * g * (
        q[1] * ca.cos(q[0]) - h * ca.sin(q[0])
      )
      G[1,0] = m_ball * g * ca.sin(q[0])

      D = TP.zeros((2, 2))
      if self.par.ball_airdrag_coef is not None:
        D[0,0] = q[1]**2 + h**2
        D[1,0] = \
        D[0,1] = -h
        D[1,1] = 1
        D *= self.par.ball_airdrag_coef

      K = dq.T @ M @ dq / 2

    B = TP.zeros((2,1))
    B[1] = 1

    B_perp = TP.zeros((1, 2))
    B_perp[0,0] = 1

    self.D_expr = D
    self.D = ca.Function('D', [q], [self.D_expr])

    self.M_expr = M
    self.M = ca.Function('M', [q], [self.M_expr])

    self.C_expr = C
    self.C = ca.Function('C', [q, dq], [self.C_expr])

    self.G_expr = G
    self.G = ca.Function('G', [q], [self.G_expr])

    self.U_expr = U
    self.U = ca.Function('U', [q], [self.U_expr])

    self.K_expr = K
    self.K = ca.Function('K', [q, dq], [self.K_expr])

    self.B_expr = B
    self.B = ca.Function('B', [q], [self.B_expr])

    self.B_perp_expr = B_perp
    self.B_perp = ca.Function('B_perp', [q], [self.B_perp_expr])

    self.E_expr = K + U
    self.E = ca.Function('E', [q, dq], [self.E_expr])

    state = ca.vertcat(q, dq)
    self.ddq_expr = ca.pinv(M) @ (-C @ dq - D @ dq - G + B @ u)
    self.rhs_expr = ca.vertcat(dq, self.ddq_expr)
    self.rhs = ca.Function('RHS', [state, u], [self.rhs_expr])

  def get_ball_pos(self):
    theta = self.q[0]
    s = self.q[1]
    h = self.par.ball_center_displacement
    ball_pos = rotmat2d(theta) @ ca.vertcat(s, h)
    return ball_pos

  def get_elems_coords(self) -> tuple:
    theta = self.q[0]
    s = self.q[1]
    h = self.par.ball_center_displacement
    ball_pos = rotmat2d(theta) @ ca.vertcat(s, h)
    ball_orient = -s / self.par.ball_radius + theta
    beam_orient = theta
    return ball_pos, ball_orient, beam_orient

  def get_airdrag_force_mat(self) -> tuple:
    if self.par.ball_airdrag_coef is None:
      return TP.zeros((2, 2))

    ball_pos, _, _ = self.get_elems_coords()
    J = jacobian(ball_pos, self.q)
    D = -self.par.ball_airdrag_coef * J.T @ J
    return D

  def get_elems_velocities(self) -> tuple:
    ball_pos, ball_orient, beam_orient = self.get_elems_coords()
    ball_vel = jtimes(ball_pos, self.q, self.dq)
    ball_angvel = jtimes(ball_orient, self.q, self.dq)
    beam_angvel = jtimes(beam_orient, self.q, self.dq)
    return ball_vel, ball_angvel, beam_angvel

  def get_kinetic_energy(self):
    ball_vel, ball_angvel, beam_angvel = self.get_elems_velocities()
    return ball_vel.T @ ball_vel * self.par.ball_mass / 2 + \
            ball_angvel**2 * self.par.ball_intertia / 2 + \
            beam_angvel**2 * self.par.beam_inertia / 2

  def get_kinetic_energy_mat(self):
    ball_pos, ball_orient, beam_orient = self.get_elems_coords()

    J1 = jacobian(ball_pos, self.q)
    M1 = J1.T @ J1 * self.par.ball_mass

    J2 = jacobian(ball_orient, self.q)
    M2 = J2.T @ J2 * self.par.ball_intertia

    J3 = jacobian(beam_orient, self.q)
    M3 = J3.T @ J3 * self.par.beam_inertia

    return M1 + M2 + M3

  def get_potential_energy(self):
    ball_pos, _, __ = self.get_elems_coords()
    return self.par.gravity_accel * self.bar.ball_mass * ball_pos[1]
