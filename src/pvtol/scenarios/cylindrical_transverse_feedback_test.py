from scipy.integrate import solve_ivp
from transverse_dynamics.transverse_feedback import (
  TranverseFeedbackController,
  TranverseFeedbackControllerPar
)
from transverse_dynamics.cylindrical_transverse_coordinates import CylindricalTransverseCoordinates
from transverse_dynamics.transverse_dynamics import TransverseDynamics
import numpy as np
import matplotlib.pyplot as plt
from common.trajectory import Trajectory
from common.numpy_utils import (
  cont_angle,
  get_max_increasing_interval,
  map_array
)
from common.plots import set_pi_xticks, set_pi_yticks
from scipy.interpolate import BSpline
from pvtol.dynamics import PVTOLAircraftDynamics
from pvtol.scenarios.sample_data import make_sample_data
from common.lqr import lqr_ltv_periodic
from pvtol.sim import (
  PVTOLAircraftSimulator,
  PVTOLAircraftSimulatorPar,
  PVTOLAircraftFeedback,
  SimulationResult
)
from scipy.interpolate import make_interp_spline
from common.linsys import solve_gramian_mat, find_fund_mat
from pvtol.scenarios.transient_process_plots import (
  show_transient,
  plot_transverse,
  plot_linsys
)


def verify_linsys():
  data = make_sample_data('tictoc')
  transdyn = data['transverse_dynamics']
  trans_coords = data['transverse_coordinates']
  ref_traj = data['traj']

  fb_par = TranverseFeedbackControllerPar(
    Q = np.eye(5),
    R = np.eye(2),
    nsteps = 1000,
    S = np.eye(5)
  )
  fb = TranverseFeedbackController(transdyn, fb_par)
  period = trans_coords.theta_max - trans_coords.theta_min
  F = lambda t: transdyn.A_fun(t) + transdyn.B_fun(t) @ fb.Ksp(t)
  _, M = find_fund_mat(F, [0, period], max_step=1e-2)
  evals = np.linalg.eigvals(M[-1])
  print(evals)

def simulate_closed_loop_dynamics():
  data = make_sample_data('tictoc')
  transdyn = data['transverse_dynamics']
  trans_coords = data['transverse_coordinates']
  ref_traj = data['traj']

  fb_par = TranverseFeedbackControllerPar(
    Q = np.eye(5),
    R = np.eye(2),
    nsteps = 1000,
    S = np.eye(5)
  )
  fb = TranverseFeedbackController(transdyn, fb_par)

  sim_par = PVTOLAircraftSimulatorPar(
    timestep = 1e-2,
    thrust_diap = [-10, 10],
    torque_diap = [-10, 10],
  )
  sim = PVTOLAircraftSimulator(sim_par, fb)

  simtime = 2. * (ref_traj.time[-1] - ref_traj.time[0])
  x0 = np.array([0.1, -0.5, 0., 0., 0., 0.])
  simres = sim.run(x0, 0., simtime)

  fig = show_transient(ref_traj, simres.traj, simres.ctrl_state, trans_coords.usp)
  fig.savefig('fig/pvtol_phase_coords.pdf')

  fig = plot_transverse(simres, trans_coords.usp)
  fig.savefig('fig/pvtol_transverse.pdf')

  fig = plot_linsys(transdyn, fb)

  plt.show()

if __name__ == "__main__":
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
  })
  simulate_closed_loop_dynamics()
  verify_linsys()
