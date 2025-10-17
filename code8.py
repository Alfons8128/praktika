import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import praktika as pr
from praktika import Var
from functools import partial

df = pr.read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='A2:E7')
df.columns = ['fr','f1','f2','c','w']

print(df)

df2 = pd.DataFrame()
for i in range(len(df['f1'])):
    df2.loc[i, 'f1'] = min(df.loc[i, 'f1'], df.loc[i, 'f2'])
    df2.loc[i, 'f2'] = max(df.loc[i, 'f1'], df.loc[i, 'f2'])

print(df2)
f1 = Var(df2['f1'].to_numpy(), errors=0.1, name='f_1', unit='kHz')
f2 = Var(df2['f2'].to_numpy(), errors=0.1, name='f_2', unit='kHz')
c = Var(df['c'].to_numpy(), errors=0.25, name='C_N', unit='pF')
fr = (f1 + f2) / 2
fr.unit = 'kHz'
fr.name = 'f_r'
w = (2 * np.pi * fr) ** -2 * 1e8
w.unit = '10^-14 s^-2'
w.name = r'\frac{1}{w_r^2}'
print('Computed fr:', fr)
print('Computed w:', w)

print('Done table', pr.to_table(c, f1, f2, fr, w))


fit, cov  = curve_fit(pr.F.linear, c.val, w.val, sigma=w.err, absolute_sigma=True)
d_fit = np.sqrt(np.diag(cov))
a, b = unp.uarray([*fit],[*d_fit])

print('Fit parameters:', f'{a:.1ueL}, {b:.1ueL}')

fig, ax = plt.subplots()
ax.errorbar(c.val, w.val, yerr=w.err, xerr=c.err, fmt='rs', linewidth=1, markersize=3, capsize=5,label='Experiment')
ax.plot(c.val, pr.F.linear(c.val, *fit), 'k--', label='Fit')
ax.set_xlabel(c.long_name)
plt.show()
#plt.close('all')

print('All done!')