import numpy as np
from dataclasses import dataclass

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

@dataclass
class Rect:
  x1 : float
  x2 : float
  y1 : float
  y2 : float

  @property
  def width(self):
    return self.x2 - self.x1

  @property
  def height(self):
    return self.y2 - self.y1

def covering_rect(r1 : Rect, r2 : Rect) -> Rect:
  x1 = min(r1.x1, r2.x1)
  x2 = max(r1.x2, r2.x2)
  y1 = min(r1.y1, r2.y1)
  y2 = max(r1.y2, r2.y2)
  return Rect(x1, x2, y1, y2)

def inflate_rect(r : Rect, coef : float) -> Rect:
  a = (1 + coef) / 2
  b = (1 - coef) / 2
  return Rect(
    x1 = r.x1 * a + r.x2 * b,
    x2 = r.x1 * b + r.x2 * a,
    y1 = r.y1 * a + r.y2 * b,
    y2 = r.y1 * b + r.y2 * a,
  )
