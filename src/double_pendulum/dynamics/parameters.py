import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class DoublePendulumParam:
  lengths : np.ndarray # length of links
  mass_centers : np.ndarray # distance from i-th link joint to its mass center
  masses : np.ndarray # masses on links
  inertia : np.ndarray # inertia momentum of links wrt their mass centers
  actiated_joint : int
  gravity_accel : float

  def __post_init__(self):
    pass

  def todict(self):
    d = asdict(self)
    return d
