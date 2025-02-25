from transverse_dynamics.transverse_coordinates.sample_data import make_sample_data
from transverse_dynamics.transverse_coordinates.transverse_feedback import compute_linsys_mat


def main():
  data = make_sample_data()
  trans_dyn = data['trans_dyn']

  compute_linsys_mat(trans_dyn)

main()
