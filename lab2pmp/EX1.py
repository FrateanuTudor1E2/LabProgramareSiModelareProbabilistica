import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

omega_1 = stats.expon.rvs(0, 1 / 4, size=10000)
omega_2 = stats.expon.rvs(0, 1 / 6, size=10000)

M = (omega_1 + omega_2) / 2

az.plot_posterior({'Timp mecanic 1': omega_1, 'Timp mecanic 2': omega_2, 'Media': M})
plt.show()
