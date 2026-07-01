import matplotlib.pyplot as plt
import numpy as np


lam = np.linspace(0, 2 * np.pi)
eps = 0.1
s = np.sin(lam - eps)
theta = np.sin(lam + eps)
plt.plot(s, theta)
plt.plot(s[0], theta[0], 'o')
plt.axhline(0, ls='--')
plt.axvline(0, ls='--')
plt.show()