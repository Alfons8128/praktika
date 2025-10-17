import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import praktika as pr
from praktika import Var
from functools import partial
from matplotlib.lines import Line2D

rdf = pr.read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='A2:E7')
rdf2 = pr.read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='B26:F31')
rdf.columns = ['fr','f1','f2','c','w']
rdf2.columns = ['fr','f1','f2','c','w']

print(rdf)

df = pd.DataFrame()
df2 = pd.DataFrame()
for i in range(len(rdf['f1'])):
    df.loc[i, 'f1'] = min(rdf.loc[i, 'f1'], rdf.loc[i, 'f2'])
    df.loc[i, 'f2'] = max(rdf.loc[i, 'f1'], rdf.loc[i, 'f2'])

for i in range(len(rdf2['f1'])):
    df2.loc[i, 'f1'] = min(rdf2.loc[i, 'f1'], rdf2.loc[i, 'f2'])
    df2.loc[i, 'f2'] = max(rdf2.loc[i, 'f1'], rdf2.loc[i, 'f2'])

print(df)
f1 = Var(df['f1'].to_numpy(), errors=0.1, name='f_1', unit='kHz')
f2 = Var(df['f2'].to_numpy(), errors=0.1, name='f_2', unit='kHz')
c = Var(rdf['c'].to_numpy(), errors=0.25, name='C_N', unit='pF')
fr = (f1 + f2) / 2
fr.set_lname('f_r', 'kHz')
w = (2 * np.pi * fr) ** -2 * 1e8
w.set_lname('w_r^{-2}', '10^{-14}\\, s^{-2}')

f12 = Var(df2['f1'].to_numpy(), errors=0.1, name='f_1', unit='kHz')
f22 = Var(df2['f2'].to_numpy(), errors=0.1, name='f_2', unit='kHz')
c2 = Var(rdf2['c'].to_numpy(), errors=0.25, name='C_N', unit='pF')
fr2 = (f12 + f22) / 2
fr2.set_lname('f_r', 'kHz')
w2 = (2 * np.pi * fr2) ** -2 * 1e8
w2.set_lname('w_r^{-2}', '10^{-14}\\, s^{-2}')

print('Computed fr:', fr)
print('Computed w:', w)
print('Done table', pr.to_table(c, f1, f2, fr, w))
print('Done table2', pr.to_table(c2, f12, f22, fr2, w2))

fit, cov  = curve_fit(pr.F.linear, c.val, w.val, sigma=w.err, absolute_sigma=True)
d_fit = np.sqrt(np.diag(cov))
a, b = unp.uarray([*fit],[*d_fit])
print('Fit parameters:', pr.ufmt(a, 'eL'), pr.ufmt(b, 'eL'))

fit2, cov2  = curve_fit(pr.F.linear, c2.val, w2.val, sigma=w2.err, absolute_sigma=True)
d_fit2 = np.sqrt(np.diag(cov2))
a2, b2 = unp.uarray([*fit2],[*d_fit2])
print('Fit parameters:', pr.ufmt(a2, 'eL'), pr.ufmt(b2, 'eL'))

fig, ax = plt.subplots()
ax.plot(c.val, pr.F.linear(c.val, *fit), 'g:', linewidth=1.5, label='fitovaná přímka')
errorbar = False
if errorbar:
    ax.errorbar(c.val, w.val, yerr=w.err, xerr=c.err, fmt='gs', linewidth=1, markersize=5, capsize=3, label='naměřené hodnoty')
else:
    ax.scatter(c.val, w.val, marker='s', s=25, color='green', linewidth=1, label='cívka A')

ax.plot(c2.val, pr.F.linear(c2.val, *fit2), ':', color='brown', linewidth=1.5, label='fitovaná přímka')
errorbar = False
if errorbar:
    ax.errorbar(c2.val, w2.val, yerr=w2.err, xerr=c2.err, color='brown',fmt='s', linewidth=1, markersize=5, capsize=3, label='naměřené hodnoty')
else:
    ax.scatter(c2.val, w2.val, marker='o', s=25, color='brown', linewidth=1, label='cívka B')

combined_handle = Line2D([], [], color='green', marker='s', linestyle=':', label='cívka A')
combined_handle2 = Line2D([], [], color='brown', marker='o', linestyle=':', label='cívka B')

ax.legend(handles=[combined_handle, combined_handle2])
ax.set_xlabel(c.long_name)
ax.set_ylabel(w.long_name)
plt.show()
#plt.close('all')

print('All done!')