#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:35:55 2018

@author: rkmaddox
"""

import numpy as np
from expyfun.io import read_hdf5
from pandas import DataFrame
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import ttest_rel, ttest_1samp
import matplotlib.pyplot as plt
plt.ion

pc = read_hdf5('/home/maddy/Code/data_analysis/Flip_Detect/matrix1.hdf5')['data'] * 100
# sub - cond - snr

snr = np.array([0, -3, -6])
conds = ['0', '90', '180', 'static']
n_sub = pc.shape[0]

# plt.plot(snr, pc[:, 0].T, 'k')
# plt.plot(snr, pc[:, 2].T, 'r')

dpc = pc[:, :3] - pc[:, [3]]

dpc_sem = np.std(dpc, axis=0) / np.sqrt(n_sub)
dpc_mean = dpc.mean(0)

for ci in range(dpc_mean.shape[0]):
    plt.errorbar(snr, dpc_mean[ci], yerr=dpc_sem[ci], fmt='-o')

ttest_1samp(dpc.mean(2).mean(1), 0)

print(ttest_rel(dpc[:, 0].mean(-1), dpc[:, 2].mean(-1)))
print(ttest_rel(dpc[:, 0].mean(-1), dpc[:, 1].mean(-1)))
print(ttest_rel(dpc[:, 1].mean(-1), dpc[:, 2].mean(-1)))


# %% Try to do an ANOVA
an_sub, an_angle, an_snr = np.meshgrid(
        np.arange(n_sub), [0, 90, 180], snr[::-1], indexing='ij')

data_dict = dict(subj=an_sub.ravel(),
                 snr=an_snr.ravel(),
                 angle=an_angle.ravel(),
                 dpc=dpc[..., ::-1].ravel())
data = DataFrame(data_dict)

from statsmodels.graphics.factorplots import interaction_plot
interaction_plot(data.snr, data.angle, data.dpc)


def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse)) /\
        (sum(aov['sum_sq'])+mse)
    return aov


formula = 'dpc ~ C(snr) + C(angle) + C(snr):C(angle)'
formula = 'dpc ~ C(snr) + angle + C(snr):C(subj) + angle:C(subj)'
formula = 'dpc ~ 0 + snr + C(angle) + snr:C(angle) + C(subj)'
formula = 'dpc ~ C(snr) + C(angle) + C(snr):C(subj)'
model = ols(formula, data).fit()
aov_table = anova_lm(model, typ=2, robust='hc3')

# eta_squared(aov_table)
# omega_squared(aov_table)
print(aov_table)

print('\n\n================\n\n')
fig = qqplot(model.resid, line='s')


formula = 'dpc ~ C(snr) + C(angle) + C(snr):C(angle)'
#formula = 'dpc ~ C(snr) + C(angle) + C(snr):C(subj)'
import statsmodels.api as sm
import statsmodels.formula.api as smf
md = smf.mixedlm(formula, data, groups=data.subj)
mdf = md.fit()
print(mdf.summary())

#fig = qqplot(mdf.resid, line='s')
#data.to_csv('flippyface_data.csv')
