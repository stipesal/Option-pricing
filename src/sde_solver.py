import numpy as np


def eulerMaruyama(mu, sigma, s0, N, T, W, endpoint=True, chol_correlation=None):
    tau = T/N

    if np.isscalar(W):
        dW = np.sqrt(tau) * np.random.randn(N, W, np.array(s0).size)
    else:
        dW = W[1:] - W[:-1]

    if not np.isscalar(s0):
        dW = dW @ chol_correlation

    if endpoint:
        return s0 * np.prod(1 + tau*mu + sigma*dW, axis=0)

    sol = np.zeros((N + 1, dW.shape[1])).squeeze()
    sol[0] = s0

    sol[1:] = s0 * np.cumprod(1 + tau * mu + sigma * dW, axis=0).squeeze()

    return sol


def milstein(mu, sigma, s0, N, T, W, endpoint=True, chol_correlation=None):
    tau = T/N

    if np.isscalar(W):
        dW = np.sqrt(tau) * np.random.randn(N, W, np.array(s0).size)
    else:
        dW = W[1:] - W[:-1]

    if not np.isscalar(s0):
        dW = dW @ chol_correlation

    if endpoint:
        return s0 * np.prod(1 + tau*mu + sigma*dW + sigma**2/2 * (dW**2 - tau), axis=0)

    sol = np.zeros((N + 1, dW.shape[1])).squeeze()
    sol[0] = s0

    sol[1:] = s0 * np.cumprod(1 + tau*mu + sigma * dW + sigma**2/2 * (dW**2 - tau), axis=0).squeeze()

    return sol