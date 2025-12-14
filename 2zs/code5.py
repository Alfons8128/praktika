import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

import praktika as pr

def filtr(x, u0, b):
    T = 0.02
    r = 0.01 # in MOhm
    return u0 * r * (x - b) / T * (1 - np.exp(-T/(r*(x - b))))

def filtreff(x, u0, r, T):
    return u0 * np.sqrt(r*x/(2*T) * (1 - np.exp(-2*T/(r*x)))) 

df1 = pr.read_excel('uloha5/uloha5.xlsx', 'List1', 'A7:F31')
df1.columns = ['c', 'dc', 'i', 'di', 'u', 'du']
df1.sort_values(by='c', inplace=True)

a = 0
c = pr.Var(df1['c'].to_numpy()[a:], df1['dc'].to_numpy()[a:], 'C', '\\mu F')
i = pr.Var(df1['i'].to_numpy()[a:], df1['di'].to_numpy()[a:], 'I', 'mA')
u = pr.Var(df1['u'].to_numpy()[a:], df1['du'].to_numpy()[a:], 'U', 'V')

rel1 = pr.Rel(c, u, filtr)
rel1.fit(p0 = [11, -0.5]) # u0, b
x = np.linspace(min(c.val), max(c.val), 100)
print(x)
y = filtr(x, 11, -0.5)
print('00000000000000000000',rel1.coeffs)

fig1, ax1 = plt.subplots(layout='constrained')
rel1.plot_data(ax1, err=(0,0))
rel1.plot_fit(ax1, label='fitovaná závislost')
#ax1.plot(x, y, label='předpokládaná závislost', linestyle='dashed')
ax1.legend()
fig1.savefig('uloha5/fig1.png')

pr.to_table(c, u)


####################################################################
df2 = pr.read_excel('uloha5/uloha5.xlsx', 'List1', 'J2:O25')
df2.columns = ['r', 'dr', 'i', 'di', 'du', 'ddu']
r = pr.Var(df2['r'], df2['dr'], 'R_z', 'k\\Omega')
i = pr.Var(df2['i'], df2['di'], 'I_{SS}', 'mA')
u = pr.Var(df2['du'], df2['ddu'], '\\Delta U', 'V')

pr.to_table(r, i, u)

fig2, ax2 = plt.subplots(layout='constrained')
rel2 = pr.Rel(i, u, pr.F.direct)
rel2.fit()
print('2222222222222222222',rel2.coeffs)
rel2.plot_data(ax2)
rel2.plot_fit(ax2, label='fitovaná závislost $\\Delta U = 1.949 \\, I_{{SS}}$')
ax2.legend()
fig2.savefig('uloha5/fig2.png')

#plt.show()


