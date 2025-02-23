from scipy.interpolate import BSpline, make_interp_spline
import casadi as ca
import numpy as np


class MXSpline(ca.Callback):
    def __init__(self, sp : BSpline):
        ca.Callback.__init__(self)
        nctrls,*dims = sp.c.shape
        c = np.reshape(sp.c, (nctrls, -1))
        odim = np.prod(dims, dtype=int)
        self.sp = BSpline(t=sp.t, c=c, k=sp.k, extrapolate=sp.extrapolate)
        self.sp_out = (odim, 1)
        self.construct('BSpline')
        self.cached = (None, None)

    def get_name_in(self, i):
        assert i == 0
        return 'in'

    def get_name_out(self, i):
        assert i == 0
        return 'out'

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        assert i == 0
        return ca.Sparsity.dense(1, 1)

    def get_sparsity_out(self, i):
        assert i == 0
        return ca.Sparsity.dense(*self.sp_out)

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        arg = ca.MX.sym('arg')
        tmp = ca.MX.sym('tmp', *self.sp_out)
        if not hasattr(self, 'deriv'):
            dsp = self.sp.derivative()
            self.deriv = MXSpline(dsp)
        return ca.Function('BSpline', [arg, tmp], [self.deriv(arg)], inames, onames, opts)

    def eval(self, args):
        arg = float(args[0])
        if arg == self.cached[0]:
            return [self.cached[1]]
        val = ca.DM(self.sp(arg))
        self.cached = (arg, ca.DM(val))
        return [val]


def wrap_symspline(sp : BSpline):
    return MXSpline(sp)


def interp_symspline(x, y, k=3, periodic=False, **kwargs):
    R'''
        :param x: 1-d monotonic array of arguments
        :param y: N-d array of valuesd
        :param k: degree of spline
    '''
    if periodic:
        bc_type = 'periodic'
    else:
        bc_type = 'natural'
    sp = make_interp_spline(x, y, k, bc_type=bc_type, **kwargs)
    return MXSpline(sp)
