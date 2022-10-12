import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

y1 = stats.gamma.rvs(4, 0, 1 / 3, size=10000)
y2 = stats.gamma.rvs(4, 0, 1 / 2, size=10000)
y3 = stats.gamma.rvs(5, 0, 1 / 2, size=10000)
y4 = stats.gamma.rvs(5, 0, 1 / 3, size=10000)

z = stats.uniform.rvs(0, 1, size=10000)
list2 = []

var1 = stats.gamma.rvs(4, 0, 1 / 3, size=1)
var2 = stats.gamma.rvs(4, 0, 1 / 2, size=1)
var3 = stats.gamma.rvs(5, 0, 1 / 2, size=1)
var4 = stats.gamma.rvs(5, 0, 1 / 3, size=1)

for i in range(10000):
    if z[i] < var1:
        list2.append(y1[i])
    elif var1 <= z[i] < var2:
        list2.append(y2[i])
    elif var2 <= z[i] < var3:
        list2.append(y3[i])
    elif var3 <= z[i] < var4:
        list2.append(y4[i])

az.plot_posterior({'list2': list2})

print("Media este: " + str(np.mean(list2)))
print("Probabilitatea este: " + str(np.mean(y4)))

plt.show()