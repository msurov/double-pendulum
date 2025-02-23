import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(2)
A = np.random.rand(3, 3)
evals, evecs = np.linalg.eig(A)
print(evecs)

Asym = (A + A.T) / 2
evals2, evecs2 = np.linalg.eig(Asym)
print(evecs2)