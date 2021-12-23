import numpy as np


def wienerprocess(T, N, n_paths):
    tau = T / N
    W = np.zeros((N + 1, n_paths))
    W[1:] = np.sqrt(tau) * np.random.randn(N, n_paths).cumsum(axis=0)

    return W