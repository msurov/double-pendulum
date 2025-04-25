import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Union
from common.linsys import solve_mat_ivp


def lqr_lti(
            A : np.ndarray,
            B : np.ndarray,
            Q : np.ndarray,
            R : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    R"""
        Brief
        -----
        Solve the matrix Riccati algebraic equation and compute \
            feedback controller stabilizing coefficients

        Params
        ------
        A : ndarray, shape (N, N)
            linear system matrix A(t_i)
        B : ndarray, shape (N, M)
            linear system matrix B(t_i)
        Q : ndarray, shape (N, N)
            wight matrix for x
        R : ndarray, shape (M, M)
            wight matrix for u

        Returns
        -------
        K : ndarray, (M, N)
            matrix of feedback coefficients
        P : ndarray, (N, N)
            solution of Riccati equation
    """
    P = solve_continuous_are(A, B, Q, R)
    K = -np.linalg.inv(R) @ (B.T @ P)
    return K, P

def __lqr_ltv_periodic(
            t : np.ndarray,
            A_fun : np.ndarray,
            B_fun : np.ndarray,
            Q_fun : np.ndarray,
            R_fun : np.ndarray,
            **solve_ivp_args
    ) -> Tuple[np.ndarray, np.ndarray]:

    t0 = t[0]
    npts, = np.shape(t)
    n,m = B_fun(t0).shape

    assert A_fun(t0).shape == (n, n)
    assert Q_fun(t0).shape == (n, n)
    assert R_fun(t0).shape == (m, m)

    def rhs(t, P):
        A = A_fun(t)
        B = B_fun(t)
        R = R_fun(t)
        Q = Q_fun(t)
        M = B @ np.linalg.solve(R, B.T)
        AT_P = A.T @ P
        dP = -AT_P - AT_P.T + P @ M @ P - Q
        return dP

    P0 = np.zeros((n, n), float)
    mismatch = None

    for i in range(100):
        print('# periodic LQR iteration', i, ', mismatch', mismatch)
        sol = solve_mat_ivp(rhs, [t[-1], t[0]], P0, t_eval=t[::-1], **solve_ivp_args)
        if sol is None:
            return None

        P = sol[1]
        if np.allclose(P[0], P[-1], atol=1e-5, rtol=1e-5):
            break

        mismatch = np.linalg.matrix_norm(P[0] - P[-1])
        P0 = P[-1]
        p0 = np.reshape(P0, (-1,))

    P = P[::-1]
    K = np.zeros((npts, m, n))
    for i in range(npts):
        R = R_fun(t[i])
        B = B_fun(t[i])
        K[i,:,:] = -np.linalg.solve(R, B.T) @ P[i,:,:]

    P[-1] = P[0]
    K[-1] = K[0]
    return K, P

def lqr_ltv_periodic(
            t : np.ndarray,
            A : Union[np.ndarray, Callable],
            B : Union[np.ndarray, Callable],
            Q : Union[np.ndarray, Callable],
            R : Union[np.ndarray, Callable],
            **solve_ivp_args
    ) -> Tuple[np.ndarray, np.ndarray]:
    R"""
        Brief
        -----
        Solve the matrix Riccati differential equation with periodic coefficients \
                and compute feedback controller stabilizing coefficients

        Params
        ------
        t : ndarray, shape (K,)
            array of time knots
        A : ndarray, shape (K, N, N)
            linear system matrix A(t_i)
        B : ndarray, shape (K, N, M)
            linear system matrix B(t_i)
        Q : ndarray, shape (N, N)
            wight matrix for x
        R : ndarray, shape (M, M)
            wight matrix for u

        Returns
        -------
        K : ndarray, (K, M, N)
            matrix of feedback coefficients
        P : ndarray, (K, N, N)
            solution of Riccati equation
    """
    if isinstance(A, np.ndarray):
        A_fun = make_interp_spline(t, A, k=3, bc_type='periodic')
    elif isinstance(A, Callable):
        A_fun = A
    else:
        assert False, f'Incorrect type of argument A: {type(A)}'

    if isinstance(B, np.ndarray):
        B_fun = make_interp_spline(t, B, k=3, bc_type='periodic')
    elif isinstance(B, Callable):
        B_fun = B
    else:
        assert False, f'Incorrect type of argument B: {type(B)}'

    if isinstance(Q, np.ndarray):
        if Q.ndim == 3:
            Q_fun = make_interp_spline(t, Q, k=3, bc_type='periodic')
        elif Q.ndim == 2:
            Q_fun = lambda _: Q
        else:
            assert False, f'Incorrect shape of argument Q: {Q.shape}'
    elif isinstance(Q, Callable):
        Q_fun = Q
    else:
        assert False, f'Incorrect type of argument Q: {type(Q)}'

    if isinstance(R, np.ndarray):
        if R.ndim == 3:
            R_fun = make_interp_spline(t, R, k=3, bc_type='periodic')
        elif R.ndim == 2:
            R_fun = lambda _: R
        else:
            assert False, f'Incorrect shape of argument Q: {Q.shape}'
    elif isinstance(R, Callable):
        R_fun = R
    else:
        assert False, f'Incorrect type of argument Q: {type(R)}'

    return __lqr_ltv_periodic(t, A_fun, B_fun, Q_fun, R_fun, **solve_ivp_args)

def lqr_ltv(
            t : np.ndarray,
            A : np.ndarray,
            B : np.ndarray,
            Q : np.ndarray,
            R : np.ndarray,
            S : np.ndarray,
            method='euler'
    ) -> Tuple[np.ndarray, np.ndarray]:
    R"""
        Brief
        -----
        Solve the matrix Riccati differential equation with periodic coefficients \
                and compute feedback controller stabilizing coefficients
        Find control of the form u(t,x) = K(t) x that minimizes the functional
        \[
            J=\int_{0}^{T}u_{s}^{T}R_{s}u_{s}+x_{s}^{T}Q_{s}x_{s}ds+x_{T}^{T}Sx_{T}
        \]

        Params
        ------
        t : ndarray, shape (K,)
            array of time knots
        A : ndarray, shape (K, N, N)
            linear system matrix A(t_i)
        B : ndarray, shape (K, N, M)
            linear system matrix B(t_i)
        Q : ndarray, shape (N, N)
            wight matrix for x
        R : ndarray, shape (M, M)
            wight matrix for u
        S : ndarray, shape (N, N)
            weight coefficients
        method : 'euler' | 'fine'

        Returns
        -------
        K : ndarray, (K, M, N)
            matrix of feedback coefficients
        P : ndarray, (K, N, N)
            solution of Riccati equation
    """

    if method == 'euler':
        return lqr_ltv_solve_euler(t, A, B, Q, R, S)
    if method == 'fine':
        return lqr_ltv_solve_fine(t, A, B, Q, R, S)
    assert False, 'Unknown method'


def lqr_ltv_solve_fine(
            t : np.ndarray,
            A : np.ndarray,
            B : np.ndarray,
            Q : np.ndarray,
            R : np.ndarray,
            S : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    R"""
        Brief
        -----
        Solve the matrix Riccati differential equation \
                and compute feedback controller stabilizing coefficients

        Params
        ------
        t : ndarray, shape (K,)
            array of time knots
        A : ndarray, shape (K, N, N)
            linear system matrix A(t_i)
        B : ndarray, shape (K, N, M)
            linear system matrix B(t_i)
        Q : ndarray, shape (N, N)
            wight matrix for x
        R : ndarray, shape (M, M)
            wight matrix for u
        S : ndarray, shape (N, N)
            weight coefficients

        Returns
        -------
        K : ndarray, (K, M, N)
            matrix of feedback coefficients
        P : ndarray, (K, N, N)
            solution of Riccati equation
    """

    npts,n,m = B.shape
    assert A.shape == (npts,n,n)
    assert Q.shape == (npts,n,n)
    assert R.shape == (npts,m,m)
    assert S.shape == (n,n)

    fun_A = make_interp_spline(t, A, k=3)
    fun_Q = make_interp_spline(t, Q, k=3)
    inv_R = np.array([np.linalg.inv(R[i,:,:]) for i in range(npts)])
    M = np.array([B[i] @ inv_R[i] @ B[i].T for i in range(npts)])
    fun_M = make_interp_spline(t, M)

    def rhs(t, p):
        P = np.reshape(p, (n,n))
        A_ = fun_A(t)
        Q_ = fun_Q(t)
        M_ = fun_M(t)
        ATP = A_.T @ P
        dP = -ATP - ATP.T + P @ M_ @ P - Q_
        dp = np.reshape(dP, (-1,))
        return dp

    s = np.reshape(S, (-1,))
    sol = solve_ivp(rhs, [t[-1], t[0]], s, t_eval=t[::-1], max_step=1e-2)
    P = np.reshape(sol.y.T, (npts, n, n))
    P = P[::-1]

    K = np.zeros((npts, m, n))
    for i in range(npts):
        K[i] = -inv_R[i] @ B[i].T @ P[i]

    return K, P


def lqr_ltv_solve_euler(
            t : np.ndarray,
            A : np.ndarray,
            B : np.ndarray,
            Q : np.ndarray,
            R : np.ndarray,
            S : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    R"""
        Brief
        -----
        Solve the matrix Riccati differential equation \
                and compute feedback controller stabilizing coefficients

        Params
        ------
        t : ndarray, shape (K,)
            array of time knots
        A : ndarray, shape (K, N, N)
            linear system matrix A(t_i)
        B : ndarray, shape (K, N, M)
            linear system matrix B(t_i)
        Q : ndarray, shape (N, N)
            wight matrix for x
        R : ndarray, shape (M, M)
            wight matrix for u
        S : ndarray, shape (N, N)
            weight coefficients

        Returns
        -------
        K : ndarray, (K, M, N)
            matrix of feedback coefficients
        P : ndarray, (K, N, N)
            solution of Riccati equation
    """

    npts,n,m = B.shape
    assert A.shape == (npts,n,n)
    assert Q.shape == (npts,n,n)
    assert R.shape == (npts,m,m)
    assert S.shape == (n,n)

    inv_R = np.array([np.linalg.inv(R[i,:,:]) for i in range(npts)])
    M = np.array([B[i] @ inv_R[i] @ B[i].T for i in range(npts)])

    P = np.zeros((npts, n, n))
    P[-1] = S

    for i in range(npts - 1, 0, -1):
        dP = -A[i].T @ P[i] - P[i] @ A[i] + P[i] @ M[i] @ P[i] - Q[i]
        P[i-1] = P[i] - dP * (t[i] - t[i-1])

    K = np.zeros((npts, m, n))
    for i in range(npts):
        K[i] = -inv_R[i] @ B[i].T @ P[i]

    return K, P
