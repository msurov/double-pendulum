from sigproc.discretization_filter import TimeDiscretization
import matplotlib.pyplot as plt
import numpy as np

def sample():
    t = np.arange(0, 10, 0.1)
    u = np.sin(t)
    f1 = TimeDiscretization(0.5)
    f2 = TimeDiscretization(0.1)
    y1 = [f1(ti, ui) for ti,ui in zip(t, u)]
    y2 = [f2(ti, ui) for ti,ui in zip(t, u)]
    plt.plot(t, u)
    plt.plot(t, y1)
    plt.plot(t, y2)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    sample()
