import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points) # uniform prior
    #incercam cu alti prori
    #prior = (grid <= 0.5).astype(int)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('Teta')
plt.show()
# crestem cantitatea de date observate
grid, posterior = posterior_grid(points, h+11, t+16)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h+11}, tails = {t+16}')
plt.yticks([])
plt.xlabel('Teta')
plt.show()