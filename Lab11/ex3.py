import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def posterior_grid(grid_points=50, heads=6, tails=9):
    """A grid implementation for the beta-binomial problem"""
    grid = np.linspace(0, 1, grid_points)
    prior = stats.beta.pdf(grid, 2, 5) # beta distribution with alpha=2 and beta=5
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

def metropolis(func, prior, draws=10000):
    """A very simple Metropolis implementation for the beta-binomial problem"""
    trace = np.zeros(draws)
    old_x = 0.5 # func.mean()
    old_prob = func.pmf(old_x) * prior.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pmf(new_x) * prior.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace

n = 10
alpha = 2
beta = 5
func = stats.binom(n, 0.5)
prior = stats.beta(alpha, beta)
trace = metropolis(func=func, prior=prior)

data = np.repeat([0, 1], (10, 3))
points = 50
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)

plt.hist(trace, bins=25, density=True, label='Metropolis')

plt.plot(grid, posterior, 'o-', label='Grid computing')

plt.xlabel('x')
plt.ylabel('pdf(x)')
plt.yticks([])
plt.legend()
plt.show()