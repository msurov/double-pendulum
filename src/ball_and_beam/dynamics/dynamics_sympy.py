from sympy import Dummy, Matrix, sin, cos, Dummy, Derivative, symbols, latex
from ball_and_beam.dynamics import BallAndBeamPar, ball_and_beam_parameters_default


def rotmat2d(angle):
  return Matrix([
    [cos(angle), -sin(angle)],
    [sin(angle), cos(angle)],
  ])

def make_mat(arg) -> Matrix:
  if isinstance(arg, Matrix):
    return arg
  elif hasattr(arg, '__iter__'):
    return Matrix(arg)
  else:
    return Matrix([arg])

def jtimes(f, x, v):
  f = make_mat(f)
  x = make_mat(x)
  v = make_mat(v)

  assert x.shape[0] == v.shape[0], f"Dimension mismatch: x has {x.shape[0]} elements, v has {v.shape[0]}"

  t = Dummy('t', real=True)
  perturbed_vars = [xi + t * vi for xi, vi in zip(x, v)]
  subs_dict = {
    x[i]: perturbed_vars[i]
      for i in range(x.shape[0])
  }
  f_perturbed = f.subs(subs_dict)
  result = Derivative(f_perturbed, t).subs(t, 0).doit()
  return result

def jacobian(f, x):
  f = make_mat(f)
  x = make_mat(x)
  return f.jacobian(x)

def get_elems_coords(q : tuple, par : BallAndBeamPar) -> tuple:
  theta, s = q
  h = par.ball_center_displacement
  ball_pos = rotmat2d(theta) @ Matrix([s, h])
  ball_orient = -s / par.ball_radius + theta
  beam_orient = theta
  return ball_pos, ball_orient, beam_orient

def get_elems_velocities(q : tuple, dq : tuple, par : BallAndBeamPar) -> tuple:
  ball_pos, ball_orient, beam_orient = get_elems_coords(q, par)
  q = make_mat(q)
  dq = make_mat(dq)
  ball_vel = jtimes(ball_pos, q, dq)
  ball_angvel = jtimes(ball_orient, q, dq)
  beam_angvel = jtimes(beam_orient, q, dq)
  return ball_vel, ball_angvel[0,0], beam_angvel[0,0]

def get_kinetic_energy(q : tuple, dq : tuple, par : BallAndBeamPar):
  ball_vel, ball_angvel, beam_angvel = get_elems_velocities(q, dq, par)
  return ball_vel.dot(ball_vel) * par.ball_mass / 2 + \
          ball_angvel**2 * par.ball_intertia / 2 + \
          beam_angvel**2 * par.beam_inertia / 2

def get_kinetic_energy_mat(q : tuple, par : BallAndBeamPar):
  ball_pos, ball_orient, beam_orient = get_elems_coords(q, par)

  J1 = jacobian(ball_pos, q)
  M1 = J1.T @ J1 * par.ball_mass

  J2 = jacobian(ball_orient, q)
  M2 = J2.T @ J2 * par.ball_intertia

  J3 = jacobian(beam_orient, q)
  M3 = J3.T @ J3 * par.beam_inertia

  return M1 + M2 + M3

def main():
  q = symbols('theta s', real=True)
  dq = symbols('dtheta ds', real=True)

  J_ball = symbols('J_ball', real=True, positive=True)
  R_ball = symbols('R_ball', real=True, positive=True)
  J_beam = symbols('J_beam', real=True, positive=True)
  h = symbols('h', real=True, positive=True)
  g = symbols('g', real=True, positive=True)
  m_ball = symbols('m_ball', real=True, positive=True)

  par = BallAndBeamPar(
    gravity_accel = g,
    beam_inertia = J_beam,
    ball_intertia = J_ball,
    ball_center_displacement = h,
    ball_mass = m_ball,
    ball_radius = R_ball,
    ball_rolling_friction_coef = 0,
  )

  M = get_kinetic_energy_mat(q, par)
  M.simplify()
  print(latex(M))

if __name__ == '__main__':
  main()
