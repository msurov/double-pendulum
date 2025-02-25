from sigproc.delay_filter import DelayFilter
import numpy as np


def test_delay_filter():
    delay = DelayFilter(0)
    x = list(range(10))
    y = [delay(i, i) for i in x]
    assert np.allclose(x, y)

    delay = DelayFilter(1)
    y = [delay(i, i) for i in x]
    assert np.allclose(x[:-1], y[1:])

    delay = DelayFilter(3)
    y = [delay(i, i) for i in x]
    assert np.allclose(x[:-3], y[3:])
