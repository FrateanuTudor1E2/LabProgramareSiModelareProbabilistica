import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from numpy import random

fastfood = np.random.poisson(20, 10)
less_15 = list()
alphaT = 0
client = 0

for i in range(len(fastfood)):
    model = pm.Model()

    with model:
        comanda = pm.Normal('C', sigma=0.5, mu=1)
        gatire = pm.Exponential('G', 1 / 4.6)

    trace = pm.sample(2000, chains=1, model=model)

    dictionary = {
        'comanda': trace['C'].tolist(),
        'gatire': trace['G'].tolist()
    }

    df = pd.DataFrame(dictionary)
    copy_df = df.copy()

    timp = df['comanda'] + df['gatire'] < 15
    less_15.extend(timp)
    alpha = np.mean(less_15)

    client = (df['comanda'] + df['gatire'] < 15)/df.shape[0]

print(np.mean(less_15))
print(np.mean(client))