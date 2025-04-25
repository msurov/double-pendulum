from pvtol.dynamics import (
  PVTOLAircraftDynamics
)
import casadi as ca
import numpy as np
from scipy.integrate import ode
from common.trajectory import Trajectory
from copy import copy
from dataclasses import dataclass
from typing import Callable, List


class PVTOLAircraftFeedback:
  def __call__(self, time : float, phase : np.ndarray) -> float:
    R"""
      compute control value
    """
    assert False, 'Not implemented'

@dataclass
class SimulationResult:
  traj : Trajectory
  ctrl_state : np.ndarray

@dataclass
class PVTOLAircraftSimulatorPar:
  timestep : float
  thrust_diap : List[float]
  torque_diap : List[float]

class PVTOLAircraftSimulator:
  R"""
      Control system simulator
  """
  def __init__(self,
        par : PVTOLAircraftSimulatorPar,
        fb : PVTOLAircraftFeedback
    ):
    self.dynamics = PVTOLAircraftDynamics()
    self.fb = fb
    self.step = par.timestep

    self.control_min = np.array([
      par.thrust_diap[0], par.torque_diap[0]
    ])
    self.control_max = np.array([
      par.thrust_diap[1], par.torque_diap[1]
    ])

    self.dynamics_state = None
    self.time = None

  def __init_signals(self, time : float, initial_state : np.ndarray):
    dynamics_state = np.reshape(initial_state, (6,))
    dynamics_state = np.copy(dynamics_state)
    self.dynamics_state = dynamics_state
    self.time = time

  def __init_control(self):
    u = self.fb(self.time, self.dynamics_state)
    u = np.clip(u, self.control_min, self.control_max)
    self.u = np.array(u)

  def __update_control(self):
    u = self.fb(self.time, self.dynamics_state)
    u = np.clip(u, self.control_min, self.control_max)
    self.u = np.array(u)

  def run(self, initial_state : np.ndarray, tstart : float = 0., tend : float = 10.) -> SimulationResult:

    self.__init_signals(tstart, initial_state)
    self.__init_control()

    rhs = lambda _, x: self.dynamics.rhs(x, self.u)
    integrator = ode(rhs)
    integrator.set_initial_value(self.dynamics_state, self.time)
    integrator.set_integrator('dopri5', max_step=self.step)

    solt = [self.time]
    solx = [self.dynamics_state.copy()]
    solu = [self.u.copy()]

    if hasattr(self.fb, 'state'):
      ctrl_state = [copy(self.fb.state)]
    else:
      ctrl_state = None

    while self.time < tend:
      if not integrator.successful():
        print('[warn] integrator doesn\'t feel good')

      # step of integration
      integrator.integrate(self.time + self.step)
      self.time = integrator.t
      self.dynamics_state = integrator.y
      self.__update_control()

      solt.append(self.time)
      solx.append(self.dynamics_state.copy())
      solu.append(self.u)

      if hasattr(self.fb, 'state'):
        ctrl_state.append(copy(self.fb.state))
    
    if ctrl_state is not None:
      ctrl_state = np.array(ctrl_state)

    result = SimulationResult(
      traj = Trajectory(
        time = np.reshape(solt, (-1,)),
        phase = np.reshape(solx, (-1, 6)),
        control = np.reshape(solu, (-1, 2)),
      ),
      ctrl_state = ctrl_state
    )
    return result
