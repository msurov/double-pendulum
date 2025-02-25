import numpy as np
from copy import copy


class DelayFilter:
    def __init__(self, nsteps : int):
        assert isinstance(nsteps, int)

        self.buf = None
        self.bufsz = nsteps + 1

    def set_initial_value(self, t0, sig0):
        self.buf = [(t0, np.copy(sig0))] * self.bufsz

    def update(self, t, sig):
        if self.buf is None:
            self.buf = [(t, np.copy(sig))] * self.bufsz

        self.buf.append((t, np.copy(sig)))
        n = max(len(self.buf) - self.bufsz, 0)
        self.buf = self.buf[n:]
        self.value = self.buf[0][1]

    def __call__(self, t, sig):
        self.update(t, sig)
        return self.value

