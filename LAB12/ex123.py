import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

#ex1
data1 = az.load_arviz_data("centered_eight")
az.summary(data1)  #number os chains and sample size

# Plot the posterior distribution of the mean
az.plot_posterior(data1)
az.plot_trace(data1)
plt.show()

data2 = az.load_arviz_data("non_centered_eight")
az.summary(data2)  #number os chains and sample size

# Plot the posterior distribution of the mean
az.plot_posterior(data2)
az.plot_trace(data2)
plt.show()

###########
# ex2

print(az.compare({'centered': data1, 'non-centered': data2},
           method='BB-ELBO',
           ic='LOO-CESS',
           parallel=True,
           waic_scale='deviance',
           var_names=['mu', 'tau']))

##########
#ex3
# count the number of divergences
n_divergences = data1.sample_stats.diverging.sum()
print(f"Number of divergences: {n_divergences}")

# visualize the divergences in the parameter space
az.plot_pair(data1, divergences=True)
az.plot_parallel(data1, divergences=True)
plt.show()

# count the number of divergences
n_divergences = data2.sample_stats.diverging.sum()
print(f"Number of divergences: {n_divergences}")

# visualize the divergences in the parameter space
az.plot_pair(data2, divergences=True)
az.plot_parallel(data2, divergences=True)
plt.show()


