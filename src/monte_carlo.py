import numpy as np

from src.stochastic_process import wienerprocess
from src.sde_solver import eulerMaruyama, milstein


CHUNK_SIZE = 10 ** 5


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


class MLMC:
    def __init__(self, M, eps):
        self.M = M
        self.eps = eps


    def fit(self, option_params, solver):
        self.option_params = option_params
        if solver == "EM":
            self.solver = eulerMaruyama
        elif solver == "MIL":
            self.solver = milstein


        self.s0 = option_params["s0"]
        self.r = option_params["r"]
        self.K = option_params["K"]
        self.sigma = option_params["sigma"]
        self.T = option_params["T"]
        self.option_type = option_params["option_type"]
        self.chol_correlation = None

        if self.option_type == 'call':
            self.payoff = lambda s: np.maximum(s - self.K, 0)
        elif self.option_type == 'put':
            self.payoff = lambda s: np.maximum(self.K - s, 0)
        elif self.option_type == 'basket_call':
            self.c = option_params["c"]
            self.chol_correlation = option_params["L"]
            self.payoff = lambda s: np.maximum(self.c @ s - self.K, 0)


        self.L = 0
        self.m_L = 10 ** 4
        self.sample_sizes = []
        self.estSums = np.zeros(self.L)
        self.estSumsSq = np.zeros(self.L)

        while (self.L <= 2 or not self.converged()):
            self.sample_sizes += [self.m_L]

            self.estSums = np.append(self.estSums, np.array(0))
            self.estSumsSq = np.append(self.estSumsSq, np.array(0))

            self.estSums[self.L], self.estSumsSq[self.L] = self.level_l_est(self.L, self.sample_sizes[self.L])

            var = self.estSumsSq / self.sample_sizes - (self.estSums / self.sample_sizes)**2
            stepsizes = self.T / self.M ** np.arange(self.L + 1)
            lower_bound = np.ceil(2 * self.eps ** -2.0 * np.sum(np.sqrt(var / stepsizes)) * np.sqrt(var * stepsizes))
            differences = np.maximum(self.sample_sizes, lower_bound) - self.sample_sizes
            self.sample_sizes = list(map(int, self.sample_sizes + differences))

            for l in range(self.L + 1):
                if differences[l] > 0:
                    estSum, estSumSq = self.level_l_est(l, int(differences[l]))
                    self.estSums[l] += estSum
                    self.estSumsSq[l] += estSumSq

            self.means = self.estSums / self.sample_sizes
            self.L += 1

        return np.sum(self.means), list(map(int, lower_bound))


    def level_l_est(self, l, m_l):
        if m_l > CHUNK_SIZE:
            n_iters, rem, m_l = m_l // CHUNK_SIZE, m_l % CHUNK_SIZE, CHUNK_SIZE
            if rem > 0:
                n_iters += 1
        else:
            n_iters, rem = 1, 0

        d = np.array(self.s0).size

        if l > 0:
            num_sols = np.empty((2, m_l, d)).squeeze()

        estSum = 0
        estSumSq = 0
        for i in range(n_iters):
            if i == n_iters - 1 and rem > 0:
                m_l = rem
                if l > 0:
                    num_sols = np.empty((2, m_l, d)).squeeze()

            if l == 0:
                num_sols = self.solver(self.r, self.sigma, self.s0, self.M ** l, self.T, m_l, chol_correlation=self.chol_correlation)
                opt_vals = self.payoff(num_sols) if d == 1 else self.payoff(num_sols.T)
                estSum += np.sum(opt_vals)
                estSumSq += np.sum(opt_vals ** 2)
                continue

            else:
                W = wienerprocess(self.T, self.M ** l, d * m_l).reshape(self.M ** l + 1, m_l, d).squeeze()
                t_eval = np.arange(0, self.M ** l + 1, self.M)

                num_sols[0] = self.solver(self.r, self.sigma, self.s0, self.M ** (l - 1), self.T, W[t_eval], chol_correlation=self.chol_correlation)
                num_sols[1] = self.solver(self.r, self.sigma, self.s0, self.M ** l, self.T, W, chol_correlation=self.chol_correlation)

                opt_vals = self.payoff(num_sols.reshape(2, d, m_l).squeeze())

                estSum += np.sum(opt_vals[1] - opt_vals[0])
                estSumSq += np.sum((opt_vals[1] - opt_vals[0])**2)

        disc_factor = np.exp(-self.r * self.T)
        return  disc_factor * estSum,  disc_factor * estSumSq


    def converged(self):
        return np.max(
            [
                np.abs(self.means[self.L-1]),
                np.abs(self.means[self.L-2]) / self.M,
                np.abs(self.means[self.L-3]) / self.M**2,
            ]
        ) <= (self.M - 1) * self.eps / np.sqrt(2)