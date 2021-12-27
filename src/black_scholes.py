import numpy as np
from scipy.stats import norm
from numpy.linalg import solve


def BlackScholes(s0, t, K, r, sigma, T, option_type):
    d1 = (np.log(s0 / K) + (r + sigma ** 2 / 2) * (T - t)) / np.sqrt(T - t) / sigma
    d2 = (np.log(s0 / K) + (r - sigma ** 2 / 2) * (T - t)) / np.sqrt(T - t) / sigma

    if option_type == 'call':
        val = s0 * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    elif option_type == 'put':
        val = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - s0 * norm.cdf(-d1)

    return val


def assemble_black_scholes_matrix(sigma, r, h, Svec, m):
    A_m = np.diag(np.ones(m - 2), -1) + np.diag(-2 * np.ones(m - 1), 0) + np.diag(np.ones(m - 2), +1)
    A_m = sigma ** 2 / (2 * h ** 2) * np.diag(Svec ** 2) @ A_m

    B_m = np.diag(- np.ones(m - 2), -1) + np.diag(np.ones(m - 2), +1)
    B_m = r / (2 * h) * np.diag(Svec) @ B_m

    C_m = - r * np.eye(m - 1)

    return A_m + B_m + C_m


def time_integrate_black_scholes(t, w0, sigma, r, h, space, m, f, solver):
    tau = t[1] - t[0]
    sol = np.zeros((len(t), w0.shape[0]))
    sol[0] = w0

    M = assemble_black_scholes_matrix(sigma, r, h, space, m)
    I = np.eye(w0.shape[0])

    if solver == 'implicit_euler':
        for i in range(len(t)-1):
            sol[i+1] = solve(I - tau * M, sol[i])

    elif solver == 'explicit_euler':
        for i in range(len(t)-1):
            sol[i+1] = (I + tau * M) @ sol[i]

    elif solver == 'trapezoidal_rule':
        for i in range(len(t)-1):
            sol[i+1] = solve(I - (tau/2) * M, (I + (tau/2) * M) @ sol[i])

    else: print('Solver not implemented!')

    sol += f(space)

    return sol
