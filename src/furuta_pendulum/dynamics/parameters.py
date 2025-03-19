from dataclasses import dataclass
import numpy as np

"""
  A link is the rigid body.
  The self frame of the link is as follows:
    z-axis coincides with the axis of rotation
    x-axis is points to its zero orientation
  
  l1 : is the axis of rotation of the first link in world frame
  l2 : is the axis of rotation of the second link in the first link's frame
  c1 : is the position vector of the mass center if the first rigid body in its own frame
  c2 : is the position vector of the mass center if the second rigid body in its own frame
"""

@dataclass
class FurutaPendulumPar:
  link_1_mass : float
  link_2_mass : float
  link_1_length : float
  link_2_length : float
  link_1_inertia_tensor : np.ndarray
  link_2_inertia_tensor : np.ndarray
  link_1_mass_center : np.ndarray # mass center in link's frame
  link_2_mass_center : np.ndarray # mass center in link's frame
  link_1_orient : np.ndarray # orientation of the link frame wrt parent when q=0
  link_2_orient : np.ndarray # orientation of the link frame wrt parent when q=0
  joint_1_pos : np.ndarray # position of joint-1 in world frame
  joint_2_pos : np.ndarray # position of joint-2 in link-1 frame
  gravity_accel : float

furuta_pendulum_param_default = FurutaPendulumPar(
  link_1_mass = 0.100,
  link_2_mass = 0.025,
  link_1_length = 0.1,
  link_2_length = 0.2,
  link_1_inertia_tensor = np.diag([0., 0.00015, 0.00015]),
  link_2_inertia_tensor = np.diag([0., 0.00015, 0.00015]),
  link_1_mass_center = np.array([0.01, 0., 0.]),
  link_2_mass_center = np.array([0.12, 0., 0.]),
  link_1_orient = np.eye(3),
  link_2_orient = np.array([
    [0.,  0., 1.],
    [0., -1., 0.],
    [1.,  0., 0.]
  ]),
  joint_1_pos = np.array([0., 0., 0.]),
  joint_2_pos = np.array([0.1, 0., 0.]),
  gravity_accel = 9.81
)
