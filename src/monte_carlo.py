import numpy as np

from src.sde_solver import eulerMaruyama


def MCEuropean(n_paths, s0, K, r, sigma, T, option_type, N=None):
    if N is None:
        wiener_T = np.sqrt(T) * np.random.randn(n_paths)
        underlying_T = s0 * np.exp((r - sigma**2 / 2) * T + sigma * wiener_T)
    else:
        underlying_T = eulerMaruyama(r, sigma, s0, N, T, n_paths)
    
    if option_type == 'call':
        payoff = lambda s: np.maximum(s - K, 0)
    elif option_type == 'put':
        payoff = lambda s: np.maximum(K - s, 0)

    opt_vals = np.exp(-r * T) * payoff(underlying_T)
    
    val = np.mean(opt_vals)
    variance = np.var(opt_vals) / n_paths

    return val, variance