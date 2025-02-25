from double_pendulum.dynamics.dynamics_casadi import (
  DoublePendulumDynamics,
  DoublePendulumParam
)
import casadi as ca
import numpy as np
from scipy.integrate import ode
from sigproc.delay_filter import DelayFilter
from common.trajectory import Trajectory
from copy import copy
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class DoublePendulumSignals:
  theta1 : float
  theta2 : float

class DoublePendulumFeedback:
  def __call__(self, time : float, sig : DoublePendulumSignals, phase : np.ndarray) -> float:
    R"""
      compute control value
    """
    assert False, 'Not implemented'

@dataclass
class DoublePendulumSimulatorPar:
  timestep : float
  encoder_delay : float
  encoder_ppr : int
  motor_delay : float
  motor_noise : float
  motor_torque_max : float

@dataclass
class SimulationResult:
  traj : Trajectory
  ctrl_state : np.ndarray

def round_int(val):
  return int(np.round(val))

def discretize(val : float, step : float):
  return step * np.round(val / step)

class DoublePendulumSimulator:
  R"""
      Control system simulator
  """
  def __init__(self,
        dynamics_par : DoublePendulumParam,
        sim_par : DoublePendulumSimulatorPar,
        fb : DoublePendulumFeedback
    ):
    self.dynamics = DoublePendulumDynamics(dynamics_par)
    self.fb = fb
    self.step = sim_par.timestep

    self.encoder_delay = DelayFilter(round_int(sim_par.encoder_delay / self.step))
    self.encoder_step = 2 * np.pi / sim_par.encoder_ppr

    self.motor_delay = DelayFilter(round_int(sim_par.motor_delay / self.step))
    self.motor_noise = sim_par.motor_noise
    self.torque_max = sim_par.motor_torque_max

    self.system_signals = None
    self.dynamics_state = None
    self.time = None

  def __update_signals(self):
    theta12 = self.dynamics_state[0:2]
    theta12 = discretize(theta12, self.encoder_step)
    theta12 = self.encoder_delay(self.time, theta12)
    self.system_signals = DoublePendulumSignals(theta1=theta12[0], theta2=theta12[1])

  def __init_signals(self, time : float, initial_state : np.ndarray):
    dynamics_state = np.reshape(initial_state, (4,))
    dynamics_state = np.copy(dynamics_state)
    self.dynamics_state = dynamics_state
    self.time = time

    theta12 = self.dynamics_state[0:2]
    self.encoder_delay.set_initial_value(self.time, theta12)
    self.system_signals = DoublePendulumSignals(theta1=theta12[0], theta2=theta12[1])

  def __init_control(self):
    u = self.fb(self.time, self.system_signals, self.dynamics_state)
    u = np.clip(u, -self.torque_max, self.torque_max)
    self.u = np.array(u)
    self.motor_delay.set_initial_value(self.time, self.u)

  def __update_control(self):
    u = self.fb(self.time, self.system_signals, self.dynamics_state)
    u += self.motor_noise * np.random.normal()
    u = np.clip(u, -self.torque_max, self.torque_max)
    u_delayed = self.motor_delay(self.time, u)
    self.u = u_delayed

  def run(self, initial_state : np.ndarray, tstart : float, tend : float) -> SimulationResult:

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
      self.__update_signals()
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
        phase = np.reshape(solx, (-1, 4)),
        control = np.reshape(solu, (-1, 1)),
      ),
      ctrl_state = ctrl_state
    )
    return result

simulator_parameters_ideal = DoublePendulumSimulatorPar(
  timestep = 1e-3,
  encoder_delay = 0,
  encoder_ppr = 10**4,
  motor_delay = 0,
  motor_noise = 0,
  motor_torque_max = 10**3
)
simulator_parameters_default = DoublePendulumSimulatorPar(
  timestep = 1e-3,
  encoder_delay = 4e-3,
  encoder_ppr = 4096,
  motor_delay = 2e-3,
  motor_noise = 0e-2,
  motor_torque_max = 400
)
