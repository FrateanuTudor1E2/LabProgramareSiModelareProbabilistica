import numpy as np
import arviz as az
import pymc3 as pm
from sklearn.mixture import GaussianMixture

data = np.loadtxt("mix.csv")

data_2d = data.reshape(-1, 1)

with pm.Model() as model2:

    weights = pm.Dirichlet("weights", a=np.ones(2))
    means = pm.Normal("means", mu=0, sigma=10, shape=2)
    stds = pm.HalfNormal("stds", sigma=10, shape=2)
    likelihood = pm.Mixture("likelihood", w=weights,
                            comp_dists=[pm.Normal.dist(mu=means[i], sigma=stds[i]) for i in range(2)],
                            observed=data_2d)
    # Fit the model
    with model2:
        trace2 = pm.sample(1000)
    trace2 = pm.sample(1000)

idata2 = az.from_pymc3_trace(trace2)

waic2 = az.waic(idata2)
loo2 = az.loo(idata2)

with pm.Model() as model3:
    weights = pm.Dirichlet("weights", a=np.ones(3))
    means = pm.Normal("means", mu=0, sigma=10, shape=3)
    stds = pm.HalfNormal("stds", sigma=10, shape=3)
    likelihood = pm.Mixture("likelihood", w=weights,
                            comp_dists=[pm.Normal.dist(mu=means[i], sigma=stds[i]) for i in range(3)],
                            observed=data_2d)
    with model3:
        trace3 = pm.sample(1000)

idata3 = az.from_pymc3_trace(trace3)

waic3 = az.waic(idata3)
loo3 = az.loo(idata3)

# Define the model
with pm.Model() as model4:
    # Define the weights of the mixture components
    weights = pm.Dirichlet("weights", a=np.ones(4))

    # Define the means of the mixture components
    means = pm.Normal("means", mu=0, sigma=10, shape=4)

    # Define the standard deviations of the mixture components
    stds = pm.HalfNormal("stds", sigma=10, shape=4)

    # Define the likelihood of the data
    likelihood = pm.Mixture("likelihood", w=weights,
                            comp_dists=[pm.Normal.dist(mu=means[i], sigma=stds[i]) for i in range(4)],
                            observed=data_2d)

    with model4:
        trace4 = pm.sample(1000)

idata4 = az.from_pymc3_trace(trace4)
waic4 = az.waic(idata4)
loo4 = az.loo(idata4)

print("WAIC for 2-component model:", waic2)
print("WAIC for 3-component model:", waic3)
print("WAIC for 4-component model:", waic4)

print("LOO for 2-component model:", loo2)
print("LOO for 3-component model:", loo3)
print("LOO for 4-component model:", loo4)


if waic2 < waic3 and waic2 < waic4:
    print("The 2-component model has the lowest WAIC value.")
elif waic3 < waic2 and waic3 < waic4:
    print("The 3-component model has the lowest WAIC value.")
else:
    print("The 4-component model has the lowest WAIC value.")

if loo2 < loo3 and loo2 < loo4:
    print("The 2-component model has the lowest LOO value.")
elif loo3 < loo2 and loo3 < loo4:
    print("The 3-component model has the lowest LOO value.")
else:
    print("The 4-component model has the lowest LOO value.")