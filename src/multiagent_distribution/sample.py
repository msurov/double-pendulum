import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from common.numpy_utils import soft_sign

class MultiagentDynamics:
  d_min : float
  d_ref : float
  fb_coef : float

  def __init__(self, d_min : float, d_ref : float):
    self.fb_coef = 0.1
    self.d_min = d_min
    self.d_ref = d_ref
    assert self.d_ref > self.d_min

  def dynamics(self, x_arr : np.ndarray) -> np.ndarray:
    nagents, = x_arr.shape
    assert nagents >= 2
    v = np.zeros(nagents)
    v[0] = self.control(x_arr[0], None, x_arr[1])
    for i in range(1, nagents - 1):
      v[i] = self.control(x_arr[i], x_arr[i-1], x_arr[i+1])
    v[-1] = self.control(x_arr[-1], x_arr[-2], None)
    return v
  
  def __call__(self, _t : float, x : np.ndarray) -> np.ndarray:
    return self.dynamics(x)

class MultiagentControl1(MultiagentDynamics):
  def control(self, x_cur : float, x_left : float, x_right : float) -> float:
    if x_left is None:
      return 0
    
    if x_right is None:
      x_ref = x_left + self.d_ref
      return -self.fb_coef * (x_cur - x_ref)

    assert x_cur >= x_left
    assert x_cur <= x_right
    if x_right - x_cur < self.d_min:
      # go to center
      x_ref = (x_right + x_left) / 2
    else:
      # to do x_left + d_ref
      x_ref = x_left + self.d_ref
    return -self.fb_coef * (x_cur - x_ref)

class MultiagentControl2(MultiagentDynamics):
  def control(self, x_cur : float, x_left : float, x_right : float) -> float:
    if x_left is None:
      assert x_cur <= x_right
      x_ref = x_right - self.d_ref
    elif x_right is None:
      assert x_cur >= x_left
      x_ref = x_left + self.d_ref
    else:
      assert x_cur <= x_right
      assert x_cur >= x_left
      x_ref = (x_right + x_left) / 2

    return -self.fb_coef * (x_cur - x_ref)

class MultiagentControl3(MultiagentDynamics):
  def control(self, x_cur : float, x_left : float, x_right : float) -> float:
    if x_left is None:
      return 0
    
    if x_right is None:
      x_ref = x_left + self.d_ref
      return -self.fb_coef * (x_cur - x_ref)

    assert x_cur >= x_left
    assert x_cur <= x_right
    if x_right - x_cur < self.d_min:
      # go to center
      x_ref = (x_right + x_left) / 2
    else:
      # to do x_left + d_ref
      x_ref = x_left + self.d_ref
    return -self.fb_coef * soft_sign(x_cur - x_ref)

class MultiagentControl4(MultiagentDynamics):
  def control(self, x_cur : float, x_left : float, x_right : float) -> float:
    if x_left is None:
      assert x_cur <= x_right
      x_ref = x_right - self.d_ref
    elif x_right is None:
      assert x_cur >= x_left
      x_ref = x_left + self.d_ref
    else:
      assert x_cur <= x_right
      assert x_cur >= x_left
      x_ref = (x_right + x_left) / 2

    return -self.fb_coef * soft_sign(x_cur - x_ref)

def main():
  np.random.seed(0)
  x = np.random.normal(size=11)
  x.sort()
  dynamics = MultiagentControl3(0.1, 1.0)
  sol = solve_ivp(dynamics, [0, 100], x, max_step=0.01)
  x_history = sol.y.T
  x_history -= x_history[-1,0]

  plt.plot(sol.t, x_history)
  plt.grid()
  plt.show()

if __name__ == '__main__':
  main()
