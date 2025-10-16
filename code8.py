import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import praktika as pr

df = pr.read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='A2:E7')

print(df)
print(df.to_latex(index=False))
for i in range(len(df.iloc[:])):
    df.iloc[i, 1], df.iloc[i, 2] = min(df.iloc[i, [1, 2]]), max(df.iloc[i, [1, 2]])


print(df)
print(df.iloc[:,[0,3,4]])

c = df.iloc[:,3].to_numpy()
unc_c = unp.uarray(c, 0.25)
d_c = unp.std_devs(unc_c)

f = df.iloc[:,0].to_numpy()
unc_f = unp.uarray(f, 0.1)
d_f = unp.std_devs(unc_f)

unc_w = (2 * np.pi * unc_f) ** -2
w = unp.nominal_values(unc_w)
d_w = unp.std_devs(unc_w)

print(unc_c)
print(unc_f)
print(unc_w)

fit, cov  = curve_fit(pr.linear_f, c, w, sigma=d_w, absolute_sigma=True)
d_fit = np.sqrt(np.diag(cov))
a, b = unp.uarray([*fit],[*d_fit])

print('Fit parameters:', *fit)
print('Uncertainties:', *d_fit)
print('togehter:', f'{a:.eL}, {b:.eL}')

fig, ax = plt.subplots()
ax.errorbar(c, w, yerr=d_w, xerr=d_c, fmt='rs', lw=1, ms=3, label='Experiment')
ax.plot(c, pr.linear_f(c, *fit), 'b--', label='Fit')
plt.show()
#plt.close('all')