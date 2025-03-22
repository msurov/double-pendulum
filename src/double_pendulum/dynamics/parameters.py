import numpy as np
from dataclasses import dataclass, asdict
import json


@dataclass
class DoublePendulumParam:
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
    return DoublePendulumParam(**d)

def save(cfgpath : str, par : DoublePendulumParam):
  with open(cfgpath, 'w') as f:
    d = par.todict()
    json.dump(d, f)


@dataclass
class DoublePendulumParam2:
  p : np.ndarray
  actuated_joint : int
  gravity_accel : float

  def __post_init__(self):
    pass

  def todict(self):
    d = asdict(self)
    return d
  
  @property
  def p1(self): return self.p[0]

  @property
  def p2(self): return self.p[1]

  @property
  def p3(self): return self.p[2]

  @property
  def p4(self): return self.p[3]

  @property
  def p5(self): return self.p[4]

  @property
  def p6(self): return self.p[5]


def convert_parameters(par : DoublePendulumParam) -> DoublePendulumParam2:
  I1, I2 = par.inertia
  m1, m2 = par.masses
  c1, c2 = par.mass_centers
  l1, l2 = par.lengths
  p1 = I1 + I2 + c1**2 * m1 + c2**2 * m2 + l1**2 * m2
  p2 = c2 * m2 * l1
  p3 = I2 + c2**2 * m2
  p4 = c1 * m1 + l1 * m2
  p5 = c2 * m2
  return DoublePendulumParam2(
    p = [p1, p2, p3, p4, p5],
    actuated_joint = par.actuated_joint,
    gravity_accel = par.gravity_accel
  )

double_pendulum_param_default = DoublePendulumParam(
  lengths=[1., 1.],
  mass_centers=[0.5, 0.5],
  masses=[0.2, 0.2],
  inertia=[0.05, 0.05],
  actuated_joint=0,
  gravity_accel=9.81
)
