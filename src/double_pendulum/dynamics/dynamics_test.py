import casadi as ca
import numpy as np
from double_pendulum.dynamics import dynamics_casadi
from double_pendulum.dynamics import dynamics_sympy
from double_pendulum.dynamics import dynamics_sympy_v2
from double_pendulum.dynamics.parameters import (
  DoublePendulumParam,
  DoublePendulumParam2,
  double_pendulum_param_default,
  convert_parameters
)
import numpy as np


def get_test_par() -> DoublePendulumParam:
  return DoublePendulumParam(
    lengths=[1., 1.],
    mass_centers=[0.5, 0.5],
    masses=[0.2, 0.2],
    inertia=[0.02, 0.02],
    actuated_joint=0,
    gravity_accel=9.81
  )

def test_compare_dynamics():
  par = get_test_par()
  d1 = dynamics_casadi.DoublePendulumDynamics(par)
  d2 = dynamics_sympy.DoublePendulumDynamics(par)
  q_ = [-0.3, 0.4]
  dq_ = [0.1, -0.7]
  M1 = np.array(d1.M(q_), float)
  M2 = np.array(d2.M(q_), float)
  C1 = np.array(d1.C(q_, dq_), float)
  C2 = np.array(d2.C(q_, dq_), float)
  G1 = np.array(d1.G(q_), float)
  G2 = np.array(d2.G(q_), float)
  B1 = np.array(d1.B(q_), float)
  B2 = np.array(d2.B(q_), float)
  U1 = np.array(d1.U(q_), float)
  U2 = np.array(d2.U(q_), float)
  assert np.allclose(M1, M2)
  assert np.allclose(C1 @ dq_, C2 @ dq_)
  assert np.allclose(G1, G2)
  assert np.allclose(B1, B2)
  assert np.allclose(U1, U2)

def test_symbolic_dynamics():
  symdynamics = dynamics_sympy.get_symbolic_dynamics()
  assert symdynamics['M'].shape == (2,2)
  assert symdynamics['C'].shape == (2,2)
  assert symdynamics['G'].shape == (2,1)
  assert symdynamics['B'].shape == (2,1)
  assert 'gravity_accel' in symdynamics
  assert 'lengths' in symdynamics
  assert 'masses' in symdynamics
  assert 'mass_centers' in symdynamics
  assert 'inertia' in symdynamics
  assert 'U' in symdynamics
  assert 'K' in symdynamics
  
def test_compare_sympy_and_casadi():
  par1 = double_pendulum_param_default
  q = np.random.normal(size=2)
  dq = np.random.normal(size=2)

  d1 = dynamics_casadi.DoublePendulumDynamics(par1)
  M1 = d1.M(q)
  C1 = d1.C(q, dq) @ dq[:,np.newaxis]
  G1 = d1.G(q)

  par2 = convert_parameters(par1)
  d2 = dynamics_sympy_v2.DoublePendulumDynamics(par2)
  M2 = d2.M(q)
  C2 = d2.C(q, dq) @ dq[:,np.newaxis]
  G2 = d2.G(q)

  assert np.allclose(M1, M2)
  assert np.allclose(C1, C2)
  assert np.allclose(G1, G2)

def test_show_equations():
  print('---------------------------------')
  print('Double pendulum dynamics:')
  dynamics_sympy.show_equations()
  print('---------------------------------')
  
def runall():
  test_compare_dynamics()
  test_symbolic_dynamics()
  test_show_equations()
  test_compare_sympy_and_casadi()

if __name__ == '__main__':
  runall()
