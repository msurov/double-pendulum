import numpy as np
from dataclasses import dataclass, asdict
import json
from typing import Any, Optional, Generic, TypeVar
import sympy


T = TypeVar('T', float, sympy.Symbol)

@dataclass
class BallAndBeamPar(Generic[T]):
  gravity_accel : T
  beam_inertia : T
  ball_center_displacement : T
  ball_mass : T
  ball_radius : T
  ball_intertia : Optional[T] = None
  ball_rolling_friction_coef : Optional[T] = None
  ball_airdrag_coef : Optional[T] = None

  def __post_init__(self):
    if isinstance(self.gravity_accel, float):
      self.__post_init_float()
    elif isinstance(self.gravity_accel, sympy.Symbol):
      self.__post_init_sympy()
    else:
      assert False, 'Not supported field type'

  def __post_init_sympy(self):
    if self.ball_intertia is None:
      self.ball_intertia = self.ball_mass * self.ball_radius**2 * 2 / 5

  def __post_init_float(self):
    if self.ball_rolling_friction_coef is None:
      self.ball_rolling_friction_coef = 0
    assert self.ball_rolling_friction_coef >= 0

    self.beam_inertia = float(self.beam_inertia)
    assert self.beam_inertia > 0

    self.ball_radius = float(self.ball_radius)
    assert self.ball_radius > 0

    if self.ball_intertia is None:
      self.ball_intertia = self.ball_mass * self.ball_radius**2 * 2 / 5
    else:
      self.ball_intertia = float(self.ball_intertia)
    assert self.ball_intertia > 0
    
    self.gravity_accel = float(self.gravity_accel)
    assert self.gravity_accel > 0
    self.ball_center_displacement = float(self.ball_center_displacement)
    self.ball_mass = float(self.ball_mass)
    assert self.ball_mass > 0

  def todict(self) -> dict[str, Any]:
    d = asdict(self)
    return d

def load(cfgpath : str) -> BallAndBeamPar[float]:
  with open(cfgpath, 'r') as f:
    d = json.load(f)
    return BallAndBeamPar(**d)

def save(cfgpath : str, par : BallAndBeamPar) -> None:
  with open(cfgpath, 'w') as f:
    d = par.todict()
    json.dump(d, f)

ball_and_beam_parameters_default = BallAndBeamPar[float](
  gravity_accel = 9.81,
  beam_inertia = 0.12,
  ball_center_displacement = 0.15,
  ball_mass = 0.045,
  ball_radius = 0.06,
  ball_airdrag_coef = 0.3
)
