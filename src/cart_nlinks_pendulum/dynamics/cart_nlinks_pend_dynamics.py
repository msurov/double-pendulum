import casadi as ca
from typing import List
from common.mechsys import MechanicalSystem
from dataclasses import dataclass

sin = ca.sin
cos = ca.cos

@dataclass
class LinkPar:
  mass : float
  inertia : float
  mass_center_displacement : float
  length : float

@dataclass
class CartNLinksPendPar:
  gravity_accel : float
  cart_mass : float
  links : List[LinkPar]
  nlinks : int = 0
  ndof : int = 0

  def __post_init__(self):
    self.nlinks = len(self.links)
    self.ndof = self.nlinks + 1
  
  def link_mass(self, i : int) -> float:
    return self.links[i].mass

  def link_mass_center(self, i : int) -> float:
    return self.links[i].mass_center_displacement

  def link_length(self, i : int) -> float:
    return self.links[i].length

@dataclass
class RidigBody:
  coords : ca.MX
  orientation : ca.MX
  mass : float
  inertia : float

def get_bodies(q, par : CartNLinksPendPar) -> List[RidigBody]:
  p = ca.vertcat(q[0], 0)
  bodies = []
  cart = RidigBody(
    mass = par.cart_mass,
    coords = p,
    inertia = 0,
    orientation = 0
  )
  bodies.append(cart)
  angles = ca.cumsum(q[1:])

  for i,link in enumerate(par.links):
    direction = ca.vertcat(
      sin(angles[i]),
      cos(angles[i])
    )
    body = RidigBody(
      coords = p + direction * link.mass_center_displacement,
      mass = link.mass,
      inertia = link.inertia,
      orientation = angles[i]
    )
    bodies.append(body)
    p += direction * link.length

  return bodies

def compute_coriolis_mat(M, q, dq):
  C1 = ca.jtimes(M, q, dq)
  C2 = 0.5 * ca.jacobian(M @ dq, q).T
  C = C1 - C2
  C = ca.simplify(C)
  return C

class CartNLinksPendDynamics(MechanicalSystem):
  def __init__(self, par : CartNLinksPendPar):
    ndof = par.ndof
    q = ca.MX.sym('q', ndof)
    dq = ca.MX.sym('dq', ndof)
    u = ca.MX.sym('u')
    bodies = get_bodies(q, par)

    M = ca.MX.zeros(ndof, ndof)
    U = 0

    for body in bodies:
      J = ca.jacobian(body.coords, q)
      M += J.T @ J * body.mass
      J = ca.jacobian(body.orientation, q)
      M += J.T @ J * body.inertia
      U += body.coords[1] * body.mass * par.gravity_accel

    C = compute_coriolis_mat(M, q, dq)
    G = ca.jacobian(U, q).T
    B = ca.DM.zeros(par.ndof, 1)
    B[0] = 1

    self.q = q
    self.dq = dq
    self.u = u

    self.M_expr = M
    self.C_expr = C
    self.G_expr = G
    self.B_expr = B
    self.B_perp_expr = ca.DM.eye(par.nlinks + 1)[1:,:]
    self.K_expr = dq.T @ M @ dq / 2
    self.U_expr = U
    self.E_expr = self.K_expr + self.U_expr
    self.ddq_expr = ca.solve(M, -C @ dq - G + B @ u)
    self.rhs_expr = ca.vertcat(dq, self.ddq_expr)

    self.M = ca.Function('M', [q], [M])
    self.C = ca.Function('C', [q, dq], [C])
    self.G = ca.Function('G', [q], [G])
    self.B = ca.Function('B', [q], [B])
    self.U = ca.Function('U', [q], [U])
    self.K = ca.Function('K', [q, dq], [self.K_expr])
    self.E = ca.Function('E', [q, dq], [self.E_expr])
    st = ca.vertcat(q, dq)
    self.rhs = ca.Function('rhs', [st, u], [self.rhs_expr])

cart_nlinks_pend_par_default = CartNLinksPendPar(
  cart_mass = 1.,
  gravity_accel = 1.,
  links = 3 * [
    LinkPar(
      mass = 1,
      mass_center_displacement = 0.5,
      length = 1.,
      inertia = 0.05
    )
  ]
)
