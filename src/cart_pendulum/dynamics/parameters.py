import numpy as np
from dataclasses import dataclass, asdict
import json


@dataclass
class CartPendulumPar:
  pendulum_length : float
  pendulum_mass : float
  cart_mass : float
  gravity_accel : float

  def __post_init__(self):
    pass

  def todict(self):
    d = asdict(self)
    return d

def load(cfgpath : str):
  with open(cfgpath, 'r') as f:
    d = json.load(f)
    return CartPendulumPar(**d)

def save(cfgpath : str, par : CartPendulumPar):
  with open(cfgpath, 'w') as f:
    d = par.todict()
    json.dump(d, f)

cart_pendulum_param_default = CartPendulumPar(
  pendulum_length = 1.,
  pendulum_mass = 1.,
  cart_mass = 1.,
  gravity_accel = 1.
)
