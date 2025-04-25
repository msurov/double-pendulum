from scipy.integrate import solve_ivp
from transverse_dynamics.transverse_coordinates import (
  TransverseCoordinates,
  TransverseDynamics
)
import numpy as np
import matplotlib.pyplot as plt
from common.trajectory import Trajectory
from common.numpy_utils import (
  cont_angle,
  get_max_increasing_interval,
  map_array,
  min_eigval
)
from common.plots import set_pi_xticks, set_pi_yticks
from pvtol.scenarios.sample_data import make_sample_data
from common.lqr import lqr_ltv_periodic
from common.linsys import solve_gramian_mat


def plot_linsys():
  data = make_sample_data()
  transdyn = data['transverse_dynamics']
  trans_coords = data['transverse_coordinates']
  ref_traj = data['traj']

  theta = np.linspace(0, 4*np.pi, 1000)
  A = map_array(transdyn.A_fun, theta)
  B = map_array(transdyn.B_fun, theta)

  t, W, F = solve_gramian_mat(transdyn.A_fun, transdyn.B_fun, [0, 2*np.pi], max_step=1e-3)
  evs = map_array(lambda w: np.sort(np.linalg.eigvals(w)), W)

  plt.plot(t, evs)

  plt.show()

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  plot_linsys()
