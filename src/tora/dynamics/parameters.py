import numpy as np
from dataclasses import dataclass, asdict
import json


@dataclass
class TORAPar:
  pendulum_length : float
  pendulum_mass : float
  pendulum_inertia : float
  cart_mass : float
  gravity_accel : float
  damping_coef : float
  stiffness_coef : float

  def __post_init__(self):
    pass

  def todict(self):
    d = asdict(self)
    return d

def load(cfgpath : str):
  with open(cfgpath, 'r') as f:
    d = json.load(f)
    return TORAPar(**d)

def save(cfgpath : str, par : TORAPar):
  with open(cfgpath, 'w') as f:
    d = par.todict()
    json.dump(d, f)

tora_param_default = TORAPar(
  pendulum_length = 0.04,
  pendulum_mass = 1.5,
  pendulum_inertia = 0.014,
  cart_mass = 10.5,
  gravity_accel = 9.81,
  damping_coef = 5.,
  stiffness_coef = 5300.
)
