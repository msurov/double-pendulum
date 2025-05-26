from scipy.integrate import solve_ivp
import numpy as np
from typing import Tuple, Callable


class SymMatPacker:
  def __init__(self, dim : int):
    self.dim = dim
    self.iy, self.ix = np.triu_indices(dim, 0)

  def pack(self, mat : np.ndarray):
    return np.array(mat)[self.iy, self.ix]

  def unpack(self, vec : np.ndarray):
    vec = np.array(vec)
    mat = np.zeros((self.dim, self.dim))
    mat[self.iy, self.ix] = vec
    mat[self.ix, self.iy] = vec
    return mat

def solve_symmat_ivp(rhs : Callable, tspan : Tuple[float, float], X0 : np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
  n, m = np.shape(X0)
  assert np.allclose(X0, X0.T)
  assert n == m

  packer = SymMatPacker(n)

  def rhs_vec(t, st):
    X = packer.unpack(st)
    dX = rhs(t, X)
    dst = packer.pack(dX)
    return dst
  
  x0 = packer.pack(X0)
  sol = solve_ivp(rhs_vec, tspan, x0, **kwargs)
  if sol.status != 0:
    return None

  nt, = sol.t.shape
  X = np.zeros((nt, n, n))
  X[:, packer.iy, packer.ix] = sol.y.T
  X[:, packer.ix, packer.iy] = sol.y.T
  t = sol.t

  return t, X

def solve_mat_ivp(rhs : Callable, tspan : Tuple[float, float], X0 : np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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

def find_fund_mat(Afun : Callable, tspan : Tuple[float,float], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
  R"""
    find fundamental solution matrix F(t) for the linear ODE x' = A(t) x
  """
  def rhs(t, X):
    return Afun(t) @ X

  X0 = np.eye(*Afun(tspan[0]).shape)
  return solve_mat_ivp(rhs, tspan, X0, **kwargs)
  
def solve_gramian_mat(Afun : Callable, Bfun : Callable, tspan : Tuple[float,float], **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def find_closed_loop_fund_mat(
      A_fun : Callable, B_fun : Callable, 
      K_fun : Callable, tspan : Tuple[float, float], 
      **integ_params
  ) \
    -> Tuple[np.ndarray, np.ndarray]:

  def rhs(t, X):
    return (A_fun(t) + B_fun(t) @ K_fun(t)) @ X

  n,_ = np.shape(A_fun(0))
  X0 = np.eye(n)
  return solve_mat_ivp(rhs, tspan, X0, **integ_params)

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

def solve_symmat_test():
  I = np.eye(5)
  A = np.random.normal(size=(5, 5))

  def rhs(t, X):
    dX = A @ X
    return dX + dX.T

  t1, X1 = solve_symmat_ivp(rhs, [0, 1], I, max_step=1e-3)
  t2, X2 = solve_mat_ivp(rhs, [0, 1], I, max_step=1e-3, t_eval=t1)

  assert np.allclose(X1, X2)
