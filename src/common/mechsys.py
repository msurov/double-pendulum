import casadi as ca
from typing import Callable


class MechanicalSystem:
  q : ca.MX
  ' dim N '

  dq : ca.MX
  ' dim N '

  u : ca.MX
  ' dim M '

  B_perp_expr : ca.MX
  ' dim (N - M) x N '

  M_expr : ca.MX
  ' dim N x N '

  C_expr : ca.MX
  ' dim N x N '

  G_expr : ca.MX
  ' dim N x 1'

  B_expr : ca.MX
  ' dim N x M '

  U_expr : ca.MX
  ' float '

  K_expr : ca.MX
  ' float '

  E_expr : ca.MX
  ' float '

  ddq_expr : ca.MX
  ' dim N x 1 '

  rhs_expr : ca.MX
  ' dim 2N x 1'

  M : Callable[[ca.MX], ca.MX]
  ' M(q) -> float[N x N]'

  C : Callable[[ca.MX, ca.MX], ca.MX]
  ' C(q, dq) -> float[N x N]'

  G : Callable[[ca.MX], ca.MX]
  ' G(q) -> float[N x 1]'

  B : Callable[[ca.MX], ca.MX]
  ' B(q) -> float[N x M]'

  U : Callable[[ca.MX], ca.MX]
  ' U(q) -> float'

  K : Callable[[ca.MX, ca.MX], ca.MX]
  ' K(q, dq) -> float'

  E : Callable[[ca.MX, ca.MX], ca.MX]
  ' E(q, dq) -> float'

  rhs : Callable[[ca.MX, ca.MX], ca.MX]
  ' rhs(state, u) -> float[2N x 1]'

