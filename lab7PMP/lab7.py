import numpy as np
import pandas as pd
import pymc3 as pm

if __name__ == "__main__":
    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    mhz = data['Speed'].values
    hd = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

#ex1
ex1_model = pm.Model()
with ex1_model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = pm.Deterministic('mu', alpha + beta1 * mhz + beta2 * np.log(hd))
    regr_like = pm.Normal('regr_like', mu = mu, sigma = sigma, observed = price)
    step = pm.Slice()
    idata = pm.sample(100, return_inferencedata = True, cores=4, step = step)

