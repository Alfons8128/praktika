import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

import praktika as pr

df1 = pr.read_excel('uloha4/uloha4.xlsx', sheet_name='List1', cells='O11:T20', header=None)



df1.columns = ['1', '2', '3', '4', '5', '6']
print(df1)
means = []
for col in df1.columns:
    colu = df1[col].to_numpy(dtype='float64')
    mean = colu.mean()
    std = colu.std(ddof=1)
    std = np.sqrt(std**2 + 0.01**2)
    means.append(ufloat(mean, std))

for i in range(10):
    for j in range(6):
        print(f'${df1.iloc[i, j]}$', end='')
        if j < 5:
            print(' & ', end='')
        else:
            print(' \\\\')
for i in range(6):
    print(f'${means[i]:.2uL}$', end='')
    if i < 5:
        print(' & ', end='')
    else:
        print(' \\\\')


print(df1)

df2 = pr.read_excel('uloha4/uloha4.xlsx', sheet_name='List1', cells='N29:U35')
df2.columns = ['l', 'dl', 'k', 'dk', 'w', 'dw', 't', 'dt']
print([m.nominal_value for m in means], [m.std_dev for m in means])
print(df2['l'].to_numpy(dtype='float64'), np.array([m.nominal_value for m in means]))
print(type(means[0].nominal_value))
d = pr.Var(np.array([m.nominal_value for m in means]), np.array([m.std_dev for m in means]), short_name='D', unit='mm')
l = pr.Var(df2['l'].to_numpy(dtype='float64'), df2['dl'].to_numpy(dtype='float64'), short_name='l', unit='cm')
k = pr.Var(df2['k'].to_numpy(dtype='float64'), df2['dk'].to_numpy(dtype='float64'), short_name='R_K', unit='m\Omega')
w = pr.Var(df2['w'].to_numpy(dtype='float64'), df2['dw'].to_numpy(dtype='float64'), short_name='R_W', unit='m\Omega')
t = pr.Var(df2['t'].to_numpy(dtype='float64'), df2['dt'].to_numpy(dtype='float64'), short_name='R_T', unit='m\Omega')
s = np.pi * d**2 / 4
rhok = k * s / l *10
rhow = w * s / l *10
rhot = t * s / l *10
meanrho = (rhok + rhow + rhot) / 3
print('Done table rho:')
pr.to_table(k, w, t, rhok, rhow, rhot, meanrho)


print('Stredy s neistotami:')
for m in means:
    print(f'{m:.2uL}')

print(np.mean([rhok.val, rhow.val, rhot.val], axis=0))
print(np.std([rhok.val, rhow.val, rhot.val], axis=0, ddof=1))
print(np.sqrt(np.std([rhok.val, rhow.val, rhot.val], axis=0, ddof=1)[1]**2 + 0.03**2))



