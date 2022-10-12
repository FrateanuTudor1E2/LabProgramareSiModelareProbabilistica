import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

x = stats.expon.rvs(1 / 4, size=10000)
y = stats.expon.rvs(1 / 6, size=10000)
z = stats.binom.rvs(1, 0.4, size=10000)

list = []
for i in range(1, 10000):
    if z[i] == 0:
        list.append(x[i])
    elif z[i] == 1:
        list.append(y[i])

az.plot_posterior(
    {'list': list})

print("Media este: ", np.mean(list))
print("Distributia este: ", np.std(list))
print("")

plt.show()