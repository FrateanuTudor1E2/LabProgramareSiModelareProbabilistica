import matplotlib.pyplot as plt
import numpy as np
import arviz
m_age = []
ppvt = []

file = open('data.csv', 'r')
next(file)
for line in file:
    el = line.split(',')
    m_age.append(int(el[3]))
    ppvt.append(int(el[1]))

N = 400
alpha_real = np.mean(m_age)
beta_real = np.mean(ppvt)
eps_real = np.random.normal(alpha_real, 1, size=N)


x = np.random.normal(10, 1, N)
y_real = alpha_real + beta_real * x
y = y_real + eps_real

_, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(x, y)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0)
ax[0].plot(x, y_real, 'k')
arviz.plot_kde(y, ax=ax[1])
ax[1].set_xlabel('y')
plt.tight_layout()
plt.show()