from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class Trajectory:
  time : np.ndarray
  state : np.ndarray
  control : Optional[np.ndarray]

  def __post_init__(self):
    self.state = np.array(self.state, float)
    self.time = np.array(self.time, float)
    nt, _ = self.state.shape
    assert self.time.shape == (nt,)

    if self.control is not None:
      self.control = np.array(self.control, float)
      assert self.control.shape[0] == nt

  @property
  def dim(self):
    return self.state.shape[1]

  @property
  def coords(self):
    ndim = self.dim
    return self.state[:,0:ndim//2]
  
  @property
  def vels(self):
    ndim = self.dim
    return self.state[:,ndim//2:]
