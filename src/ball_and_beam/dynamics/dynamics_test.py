from ball_and_beam.dynamics import BallAndBeamDynamics, ball_and_beam_parameters_default
import numpy as np


def main():
  par = ball_and_beam_parameters_default
  par.ball_airdrag_coef = 0.1
  dynamics1 = BallAndBeamDynamics(par, auto_compute=True)
  dynamics2 = BallAndBeamDynamics(par, auto_compute=False)

  q = [2.0534, 0.2345]
  dq = [-0.3, 0.7]

  M1 = dynamics1.M(q)
  M2 = dynamics2.M(q)
  assert np.allclose(M1, M2)

  G1 = dynamics1.G(q)
  G2 = dynamics2.G(q)
  assert np.allclose(G1, G2)

  K1 = dynamics1.K(q, dq)
  K2 = dynamics2.K(q, dq)
  assert np.allclose(K1, K2)

  U1 = dynamics1.U(q)
  U2 = dynamics2.U(q)
  assert np.allclose(U1, U2)

  C1 = dynamics1.C(q, dq)
  C2 = dynamics2.C(q, dq)
  assert np.allclose(C1 @ dq, C2 @ dq)

  D1 = dynamics1.D(q)
  D2 = dynamics2.D(q)
  assert np.allclose(D1, D2)

if __name__ == '__main__':
  main()
