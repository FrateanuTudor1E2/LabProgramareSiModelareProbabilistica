#lab3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    cutremur = pm.Bernoulli('C', 0.0005)
    incendiu = pm.Bernoulli('I',0.01)
    alarma = pm.Bernoulli('A', 0.0001)

    trace = pm.sample(200000)

dictionary = {
    'cutremur': trace['U'].tolist(),
    'incendiu': trace['I'].tolist(),
    'alarma': trace['A'].tolist(),

}
df = pd.DataFrame(dictionary)

p_alarmacutremur = df[(df['cutremur']==1)&(df['alarma']==1)].shape[0] / df[df(['alarma']==1)].shape[0]
p_incendiu_fara_alarma = df[(df['incendiu']==1)&(df['alarma']==0)].shape[0] / df[df(['alarma']==0)].shape[0]

print(p_alarmacutremur)
