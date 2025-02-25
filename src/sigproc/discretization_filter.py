import numpy as np


class TimeDiscretization:
    def __init__(self, step) -> None:
        self.step = step
        self.t_prev = None

    def reset(self):
        self.t_prev = None

    def process(self, t, u):
        if self.t_prev is None:
            self.u_prev = np.copy(u)
            self.t_prev = t

        if t < self.t_prev + self.step:
            return self.u_prev
    
        self.u_prev = u
        self.t_prev = t
        return self.u_prev
    
    def __call__(self, t, u):
        return self.process(t, u)
