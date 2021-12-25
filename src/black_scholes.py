import numpy as np
from scipy.stats import norm


def BlackScholes(s0, t, K, r, sigma, T, option_type):
    d1 = (np.log(s0 / K) + (r + sigma ** 2 / 2) * (T - t)) / np.sqrt(T - t) / sigma
    d2 = (np.log(s0 / K) + (r - sigma ** 2 / 2) * (T - t)) / np.sqrt(T - t) / sigma

    if option_type == 'call':
        val = s0 * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    elif option_type == 'put':
        val = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - s0 * norm.cdf(-d1)

    return val