import casadi as ca
from typing import Callable

class MechanicalSystem:
  T = ca.GenericExpressionCommon

  q : T
  ' dim N '

  dq : T
  ' dim N '

  u : T
  ' dim M '

  B_perp_expr : T
  ' dim (N - M) x N '

  M_expr : T
  ' dim N x N '

  C_expr : T
  ' dim N x N '

  G_expr : T
  ' dim N x 1'

  B_expr : T
  ' dim N x M '

  U_expr : T
  ' float '

  K_expr : T
  ' float '

  E_expr : T
  ' float '

  ddq_expr : T
  ' dim N x 1 '

  rhs_expr : T
  ' dim 2N x 1'

  M : Callable[[T], T]
  ' M(q) -> float[N x N]'

  C : Callable[[T, T], T]
  ' C(q, dq) -> float[N x N]'

  G : Callable[[T], T]
  ' G(q) -> float[N x 1]'

  B : Callable[[T], T]
  ' B(q) -> float[N x M]'

  B_perp : Callable[[T], T]
  ' B_perp(q) -> float[M x N]'

  D : Callable[[T], T]
  ' D(q) -> float[N x N]'

  U : Callable[[T], T]
  ' U(q) -> float'

  K : Callable[[T, T], T]
  ' K(q, dq) -> float'

  E : Callable[[T, T], T]
  ' E(q, dq) -> float'

  rhs : Callable[[T, T], T]
  ' rhs(state, u) -> float[2N x 1]'
