
_1 = ['beta', 'binomial', 'bytes', 'chisquare', 'choice', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'get_state', 'gumbel', 'hypergeometric', 'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial', 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'permutation', 'poisson', 'power', 'rand', 'randint', 'randn', 'random', 'random_integers', 'random_sample', 'ranf', 'rayleigh', 'sample', 'seed', 'set_state', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf', 'Generator', 'RandomState', 'SeedSequence', 'MT19937', 'Philox', 'PCG64', 'PCG64DXSM', 'SFC64', 'default_rng', 'BitGenerator']

import numpy as np
amounts: np.ndarray = np.random.exponential(scale=5, size=10_000_000)
amounts2: np.ndarray = np.random.uniform(10, 500, size=10_000_000)
amounts3: np.ndarray = np.random.normal(100_000, 1000, size=100_000_000)
# result: np.ndarray = np.random.get_state
# result.sort()
print(amounts3[amounts3 <= 10000])
