import numpy as np
from dataclasses import dataclass, asdict
import json


@dataclass
class NLinksPendulumParam:
  lengths : np.ndarray # length of links
  mass_centers : np.ndarray # distance from i-th link joint to its mass center
  masses : np.ndarray # masses on links
  inertia : np.ndarray # inertia momentum of links wrt their mass centers
  actuated_joint : int
  gravity_accel : float

  def __post_init__(self):
    pass

  def todict(self):
    d = asdict(self)
    return d

def load(cfgpath : str):
  with open(cfgpath, 'r') as f:
    d = json.load(f)
    return NLinksPendulumParam(**d)

def save(cfgpath : str, par : NLinksPendulumParam):
  with open(cfgpath, 'w') as f:
    d = par.todict()
    json.dump(d, f)

thripple_pendulum_param_default = NLinksPendulumParam(
  lengths=[1., 1., 1.],
  mass_centers=[0.5, 0.5, 0.5],
  masses=[0.2, 0.2, 0.2],
  inertia=[0.05, 0.05, 0.05],
  actuated_joint=0,
  gravity_accel=9.81
)
