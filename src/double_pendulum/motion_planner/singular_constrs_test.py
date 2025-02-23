from double_pendulum.motion_planner.singular_constrs import get_sing_constr_at
from double_pendulum.dynamics import (
  DoublePendulumDynamics,
  DoublePendulumParam,
  double_pendulum_param_default
)
from double_pendulum.motion_planner.reduced_dynamics import ReducedDynamics
import numpy as np


def test_sing_constr():
  par = double_pendulum_param_default
  dynamics = DoublePendulumDynamics(par)
  q_sing = np.array([-2, 0.9])
  constr = get_sing_constr_at(dynamics, q_sing)
  reduced = ReducedDynamics(dynamics, constr)

  eps = 1e-8
  assert abs(reduced.alpha(0.)) < eps
  assert abs(reduced.dalpha(0.)) > eps
  sig = np.sign(reduced.dalpha(0.))
  assert reduced.beta(0.) / reduced.dalpha(0.) < -0.5
  assert sig * reduced.gamma(0.) > 0
