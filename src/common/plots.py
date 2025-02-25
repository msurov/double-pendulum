import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction

def tex_ratio_pi(r):
    if r.numerator == 0:
        return '0'
    if r.denominator in [-1, 1]:
        s = '' if r.numerator * r.denominator > 0 else '-'
        num = abs(r.numerator)
        if num == 1:
            return '$' + s + R'\pi$'
        return R'$' + s + str(num) + R'\pi$'
    s = '' if r.numerator * r.denominator > 0 else '-'
    num = abs(r.numerator)
    den = abs(r.denominator)
    return R'$' + s + R'\frac{' + str(num) + '}{' + str(den) + R'}\pi$'

def get_ticks(step : float, diap=None):
    assert isinstance(step, (int, str, float))
    r = Fraction(step)
    n1 = int(np.round(diap[0] / (np.pi * r)))
    n2 = int(np.round(diap[1] / (np.pi * r)))
    indices = np.arange(n1, n2 + 1, dtype=int)
    xlabels = [tex_ratio_pi(e) for e in indices * r]
    xticks = np.array(indices * r * np.pi, dtype=float)
    return xticks, xlabels

def set_pi_xticks(step : float, xdiap=None, **xtickspar):
    if xdiap is None:
        xdiap = plt.xlim()
    xticks, xlabels = get_ticks(step, xdiap)
    plt.xticks(xticks, xlabels, **xtickspar)

def set_pi_yticks(step : float, ydiap=None, **ytickspar):
    if ydiap is None:
        ydiap = plt.ylim()
    yticks, ylabels = get_ticks(step, ydiap)
    plt.yticks(yticks, ylabels, **ytickspar)
