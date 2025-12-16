import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import matplotlib.pyplot as plt
from sigproc.delay_filter import DelayFilter


class Simulator:
  def __init__(self, sys, fb, initial_state : np.ndarray, step : float):
    self._state = np.copy(initial_state)
    self._time = 0.
    self._step = float(step)
    self._control = fb(self._time, self._state)
    rhs = lambda t, x: sys(t, x, self._control)
    self._integrator = ode(rhs)
    self._integrator.set_initial_value(initial_state, 0.)
    self._integrator.set_integrator('dopri5', max_step=self._step)
    self._sys = sys
    self._fb = fb

  def step(self):
    self._control = self._fb(self._time, self._state)
    self._integrator.integrate(self._time + self._step)
    if not self._integrator.successful():
      return None
    self._time = self._integrator.t
    self._state = self._integrator.y
    return self._time, self._state

  def run(self, simtime):
    nsteps = int(simtime / self._step) + 1
    assert nsteps > 0

    state = np.zeros((nsteps,) + self._state.shape)
    control = np.zeros((nsteps,) + self._control.shape)
    time = np.zeros(nsteps)
    state[0] = self._state
    time[0] = self._time

    for i in range(1, nsteps):
      res = self.step()
      if res is None:
        return None
      time[i], state[i,:] = res
      control[i] = self._control

    return time, state, control

def rössler_attractor(_, state, u):
  x,y,z = state
  dx = -y - z
  dy = x + 0.2 * y + u
  dz = 0.2 + z * (x - 5.7)
  return np.array([dx, dy, dz])

class Feedback:
  def __init__(self, period : float, fbcoef : float):
    self._delay = DelayFilter(period)
    self._fbcoef = fbcoef

  def set_period(self, value):
    self._delay = DelayFilter(value)

  def set_fbcoef(self, value):
    self._fbcoef = value

  def __call__(self, t, state):
    y = state[1]
    y_delayed = self._delay(t, y)
    return self._fbcoef * (y_delayed - y)

def main():
  initial_state = np.array([5., 3., 1.])
  step = 1e-2
  fb = Feedback()
  sim = Simulator(rössler_attractor, fb, initial_state, step)
  t, x, u = sim.run(200)
  # _, axes = plt.subplots(1, 2, sharex=True)
  # axes[0].plot(x[:,0], x[:,1])
  # axes[1].plot(x[:,0], x[:,2])
  plt.plot(t, u)
  plt.show()

if __name__ == '__main__':
  main()
