import numpy as np
import matplotlib.pyplot as plt

N = 1000
num_runs = 100
errors = np.empty(num_runs)

for i in range(num_runs):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    errors[i] = error

mean_error = np.mean(errors)
std_error = np.std(errors)

plt.errorbar(N, mean_error, std_error, fmt='o')
plt.xlabel('Number of points (N)')
plt.ylabel('Error (% of true value)')
plt.show()
