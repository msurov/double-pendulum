import numpy as np
from copy import copy


class DiscreteDelayFilter:
  def __init__(self, delay_steps : int):
    assert isinstance(delay_steps, int)

    self.buf = None
    self.bufsz = delay_steps + 1

  def set_initial_value(self, t0, sig0):
    self.buf = [(t0, np.copy(sig0))] * self.bufsz

  def update(self, t, sig):
    if self.buf is None:
      self.buf = [(t, np.copy(sig))] * self.bufsz

    self.buf.append((t, np.copy(sig)))
    n = max(len(self.buf) - self.bufsz, 0)
    self.buf = self.buf[n:]
    self.value = self.buf[0][1]

  def __call__(self, t, sig):
    self.update(t, sig)
    return self.value


class DelayFilter:
  def __init__(self, delay : float):
    delay = float(delay)
    assert delay > 0, 'Strictly positive delay is expected'
    self.__time = []
    self.__values = []
    self.__delay = delay

  @property
  def delay(self):
    return self.__delay

  def __call__(self, t, u):
    assert len(self.__time) == 0 or t > self.__time[-1]

    self.__time.append(t)
    self.__values.append(u)
    t_delayed = t - self.__delay

    i = 0
    while self.__time[i] < t_delayed:
      i += 1

    if i == 0:
      return self.__values[0]
    
    assert self.__time[i-1] < t_delayed <= self.__time[i]
    
    self.__time = self.__time[i-1:]
    self.__values = self.__values[i-1:]

    t1 = self.__time[0]
    y1 = self.__values[0]
    t2 = self.__time[1]
    y2 = self.__values[1]

    y = (t2 - t_delayed) * y1 / (t2 - t1) + (t_delayed - t1) * y2 / (t2 - t1)
    return y
