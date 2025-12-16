from sigproc.delay_filter import DiscreteDelayFilter, DelayFilter
import numpy as np


def test_discrete_delay_filter():
  delay = DiscreteDelayFilter(0)
  x = list(range(10))
  y = [delay(i, i) for i in x]
  assert np.allclose(x, y)

  delay = DiscreteDelayFilter(1)
  y = [delay(i, i) for i in x]
  assert np.allclose(x[:-1], y[1:])

  delay = DiscreteDelayFilter(3)
  y = [delay(i, i) for i in x]
  assert np.allclose(x[:-3], y[3:])

def test_delay_filter1():
  t = np.arange(0, 2, 0.03)
  u = np.sin(t)

  f = DelayFilter(0.09)
  y = np.array([f(*e) for e in zip(t,u)])

  assert np.allclose(u[:-3], y[3:])

def test_delay_filter2():
  step = 0.03
  t = np.arange(0, 10, 0.03)
  u = np.sin(t)

  delay = 0.14
  f = DelayFilter(delay)
  y = np.array([f(*e) for e in zip(t,u)])
  y_expected = np.sin(t - delay)

  i = int((delay + step - 1) / step)
  assert np.allclose(y_expected[i:], y[i:], atol=1e-4)
