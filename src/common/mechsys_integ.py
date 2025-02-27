from .mechsys import MechanicalSystem
from .trajectory import Trajectory
from typing import Callable, Optional
from scipy.integrate import solve_ivp
import numpy as np
from dataclasses import dataclass


@dataclass
class MechanicalSystemTrajectory(Trajectory):
  energy : Optional[np.ndarray] = None
  power : Optional[np.ndarray] = None

ControlInput = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


def integrate(
    sys : MechanicalSystem,
    q0 : np.ndarray,
    dq0 : np.ndarray,
    integtime : float,
    control_input : Optional[ControlInput] = None,
    **integ_args
  ) -> MechanicalSystemTrajectory:

  udim,_ = sys.u.shape
  qdim,_ = sys.q.shape

  if control_input is None:
    control_input = lambda *_: np.zeros(udim)

  def rhs(t, st):
    q = st[0:qdim]
    dq = st[qdim:]
    u = control_input(t, q, dq)
    dst = sys.rhs(st, u)
    dst = np.reshape(dst, (2*qdim,))
    return dst
  
  st0 = np.concatenate((q0, dq0))
  sol = solve_ivp(rhs, [0, integtime], st0, **integ_args)
  t = sol.t
  q = sol.y[0:qdim].T
  dq = sol.y[qdim:].T
  npts, = t.shape
  u = np.array([control_input(t[i], q[i], dq[i]) for i in range(npts)])
  u = np.reshape(u, (npts, udim))
  E = np.reshape([sys.E(q[i], dq[i]) for i in range(npts)], (npts,))
  W = np.reshape([(dq[i,np.newaxis,:] @ sys.B(q[i]) @ u[i]) for i in range(npts)], (npts,))

  return MechanicalSystemTrajectory(
    time = t,
    phase = sol.y.T,
    control = u,
    energy = E,
    power = W
  )
