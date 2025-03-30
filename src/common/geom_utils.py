import numpy as np

def enlarge_rect(r : np.ndarray, coef) -> np.ndarray:
  R"""
    @param r = [
      [x1, x2],
      [y1, y2],
    ]
  """
  c = np.mean(r, axis=1)
  w = r[:,1] - r[:,0]
  return np.array([c - w * coef / 2, c + w * coef / 2]).T
