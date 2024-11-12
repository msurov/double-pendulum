from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from fractions import Fraction as frac
import matplotlib.pyplot as plt


def pi_axis_formatter(val, pos, denomlim=, pi=R'\pi'):
  """
    format label properly
    for example: 0.6666 pi --> 2π/3
                : 0      pi --> 0
                : 0.50   pi --> π/2  
  """

  minus = "-" if val < 0 else ""
  val = abs(val)
  ratio = frac(val / np.pi).limit_denominator(denomlim)
  n, d = ratio.numerator, ratio.denominator
  print(val, n, d)
  
  fmt2 = "%s" % d 
  if n == 0:
    fmt1 = "0"
  elif n == 1:
    fmt1 = pi
  else:
    fmt1 = r"%s%s" % (n,pi)

  fmtstring = "$" + minus + (fmt1 if d == 1 else r"{%s}/{%s}" % (fmt1, fmt2)) + "$"  
  return fmtstring

def set_xaxis_pi_ticks(ax : plt.Axes):
  ax.xaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))

def set_yaxis_pi_ticks(ax : plt.Axes):
  ax.yaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
