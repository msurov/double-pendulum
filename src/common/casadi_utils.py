import casadi as ca


def rot_x(angle):
  return rotmat(angle, ca.DM([1, 0, 0]))

def rot_y(angle):
  return rotmat(angle, ca.DM([0, 1, 0]))

def rot_z(angle):
  return rotmat(angle, ca.DM([0, 0, 1]))

def normalized(v):
  return v / ca.norm_2(v)

def rotmat(angle, axis):
  I = ca.DM_eye(3)
  K = wedge(normalized(axis))
  R = I + ca.sin(angle) * K + (1 - ca.cos(angle)) * K @ K
  return ca.simplify(R)

def wedge(v):
  x = v[0]
  y = v[1]
  z = v[2]
  return ca.vertcat(
    ca.horzcat(0, -z, y),
    ca.horzcat(z, 0, -x),
    ca.horzcat(-y, x, 0)
  )

def vee(A):
  return ca.vertcat(A[2,1], A[0,2], A[1,0])

def compute_angvel_jac(R, x):
  R"""
    For the rotation matrix R(x) find expression for self-frame angular velocity w as
      w = S(x) x'
    The function computes the matrix S(x)
  """
  r = ca.reshape(R, (9, 1))
  n,m = x.shape
  assert m == 1
  dr = ca.jacobian(r, x)
  jac = ca.horzcat(
    *[vee(R.T @ ca.reshape(dr[:,i], (3, 3))) for i in range(n)]
  )
  jac = ca.simplify(jac)
  return jac
