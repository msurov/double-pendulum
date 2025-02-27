from furuta_pendulum.dynamics import (FurutaPendulumDynamics, furuta_pendulum_param_default)
from common.mechsys_integ import integrate
import numpy as np
import matplotlib.pyplot as plt
from common.numpy_utils import integrate_array


def test_energy_conserve():
  fb = lambda _, q, __: np.array([-q[0]])
  dynamics = FurutaPendulumDynamics(furuta_pendulum_param_default)
  q0 = np.zeros(2)
  dq0 = 1e-2 * np.ones(2)
  traj = integrate(dynamics, q0, dq0, 5, fb, max_step=1e-3)
  dE = integrate_array(traj.time, traj.power)
  assert np.allclose(traj.energy - dE, traj.energy[0])
