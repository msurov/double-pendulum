import casadi as ca
import numpy as np
from . import dynamics_casadi
from . import dynamics_sympy


def get_test_par() -> dynamics_casadi.DoublePendulumParam:
  return dynamics_casadi.DoublePendulumParam(
    lengths=[1., 1.],
    mass_centers=[0.5, 0.5],
    masses=[0.2, 0.2],
    inertia=[0.02, 0.02],
    actiated_joint=0,
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
  assert np.allclose(C1, C2)
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
  
def test_show_equations():
  print('---------------------------------')
  print('Double pendulum dynamics:')
  dynamics_sympy.show_equations()
  print('---------------------------------')
  
def runall():
  test_compare_dynamics()
  test_symbolic_dynamics()
  test_show_equations()

if __name__ == '__main__':
  runall()