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
raw_c = df.iloc[:,3].to_numpy()
c = unp.uarray(raw_c, 0.25)
raw_f = df.iloc[:,0].to_numpy()
f = unp.uarray(raw_f, 0.1)
w = df.iloc[:,4].to_numpy()
print(c)
print(f)

fit, cov  = curve_fit(pr.linear_f, unp.nominal_values(c), unp.nominal_values(w), sigma=unp.errors(w), absolute_sigma=True)
print('Fit parameters:', fit)
print('Covariance matrix:\n', cov)
da = np.sqrt(cov[0][0])
db = np.sqrt(cov[1][1])

fig, ax = plt.subplots()
ax.errorbar(unp.nominal_values(c), unp.nominal_values(w), yerr=unp.errors(w), xerr=unp.errors(c), fmt='rs', lw=1, ms=3, label='Experiment')
ax.plot(unp.nominal_values(c), pr.linear_f(unp.nominal_values(c), *fit), 'b--', label='Fit')

plt.close('all')