from scipy.integrate import solve_ivp
import numpy as np
from typing import Tuple, Callable


def solve_mat_ivp(rhs : Callable, tspan : Tuple[float], X0 : np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
  n, m = np.shape(X0)

  def rhs_vec(t, st):
    X = np.reshape(st, (n, m))
    dX = rhs(t, X)
    dst = np.reshape(dX, (n*m,))
    return dst
  
  x0 = np.reshape(X0, (n*m,))
  sol = solve_ivp(rhs_vec, tspan, x0, **kwargs)
  if sol.status != 0:
    return None

  t = sol.t
  X = np.reshape(sol.y.T, (-1, n, m))
  return t, X

def find_fund_mat(Afun : Callable, tspan : Tuple[float], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
  R"""
    find fundamental solution matrix F(t) for the linear ODE x' = A(t) x
  """
  def rhs(t, X):
    return Afun(t) @ X

  X0 = np.eye(*Afun(tspan[0]).shape)
  return solve_mat_ivp(rhs, tspan, X0, **kwargs)
  
def solve_gramian_mat(Afun : Callable, Bfun : Callable, tspan : Tuple[float], **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  n = Afun(tspan[0]).shape[0]

  def rhs(t, WF):
    W = WF[0:n,0:n]
    F = WF[n:2*n,0:n]
    Finv = np.linalg.inv(F)
    A = Afun(t)
    B = Bfun(t)
    dW = Finv @ B @ B.T @ Finv.T
    dF = A @ F
    return np.concatenate((dW, dF))

  WF0 = np.zeros((2*n, n))
  WF0[0:n,0:n] = 0
  WF0[n:2*n,0:n] = np.eye(n)

  t, WF = solve_mat_ivp(rhs, tspan, WF0, **kwargs)
  W = WF[:,0:n,0:n]
  F = WF[:,n:2*n,0:n]

  return t, W, F

def solve_fund_mat_test():
  def Afun(t):
    return np.array([
      [0, -1],
      [1, 0]
    ])
  t, X = find_fund_mat(Afun, [0, 2*np.pi], max_step=1e-2)
  assert np.allclose(X[:,0,0], +np.cos(t))
  assert np.allclose(X[:,1,0], +np.sin(t))
  assert np.allclose(X[:,0,1], -np.sin(t))
  assert np.allclose(X[:,1,1], +np.cos(t))
