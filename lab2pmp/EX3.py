import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
x = stats.expon.rvs(0.3, size=10)
y = stats.expon.rvs(0.7, size=10)

z = stats.binom.rvs(1, 0.3, size=10)
list3 = []
list4 = []
list5 = []
list6 = []

matrice_Prob = []

for i in range(10):
    for j in range(10):
        if z[i] == 0 and z[j] == 0:
            list3.append(x[i])
            list3.append(x[i])
matrice_Prob.append(list3)

z = stats.binom.rvs(1, 0.3, size=10)

for i in range(10):
    for j in range(10):
        if z[i] == 0 and z[j] == 1:
            list4.append(x[i])
            list4.append(y[i])
matrice_Prob.append(list4)

z = stats.binom.rvs(1, 0.3, size=10)

for i in range(10):
    for j in range(10):
        if z[i] == 1 and z[j] == 0:
            list5.append(y[i])
            list5.append(x[i])
matrice_Prob.append(list5)

z = stats.binom.rvs(1, 0.3, size=10)

for i in range(10):
    for j in range(10):
        if z[i] == 1 and z[j] == 1:
            list6.append(y[i])
            list6.append(y[i])
matrice_Prob.append(list6)

az.plot_posterior({'Moneda1': list3})
az.plot_posterior({'Moneda2': list4})
az.plot_posterior({'Moneda3': list5})
az.plot_posterior({'Moneda4': list6})

print("Media este: " + str(np.mean(matrice_Prob)))

plt.show()