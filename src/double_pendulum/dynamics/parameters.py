import numpy as np
from dataclasses import dataclass, asdict
import json


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

def load(cfgpath : str):
  with open(cfgpath, 'r') as f:
    d = json.load(f)
    return DoublePendulumParam(**d)

def save(cfgpath : str, par : DoublePendulumParam):
  with open(cfgpath, 'w') as f:
    d = par.todict()
    json.dump(d, f)


@dataclass
class DoublePendulumParam2:
  p : np.ndarray
  actiated_joint : int
  gravity_accel : float

  def __post_init__(self):
    pass

  def todict(self):
    d = asdict(self)
    return d


double_pendulum_param_default = DoublePendulumParam(
  lengths=[1., 1.],
  mass_centers=[0.5, 0.5],
  masses=[0.2, 0.2],
  inertia=[0.05, 0.05],
  actiated_joint=0,
  gravity_accel=9.81
)
