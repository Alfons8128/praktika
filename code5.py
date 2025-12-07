import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

import praktika as pr

def osciloskop_eq(x, u0):
    return None

df1 = pr.read_excel('uloha5/uloha5.xlsx', 'List1', 'A7:F31')
df1.columns = ['c', 'dc', 'i', 'di', 'u', 'du']
df1.sort_values(by='c', inplace=True)

c = pr.Var(df1['c'].to_numpy(), df1['dc'].to_numpy(), 'C', '\\mu F')
i = pr.Var(df1['i'].to_numpy(), df1['di'].to_numpy(), 'I', 'mA')
u = pr.Var(df1['u'].to_numpy(), df1['du'].to_numpy(), 'U', 'V')

rel1 = pr.Rel(c, u, osciloskop_eq)
#rel1.fit()

fix1, ax1 = plt.subplots(layout='constrained')
rel1.plot_data(ax1, err=(0,0))
#rel1.plot_fit(ax1, label='fitovaná závislost')
ax1.legend()

pr.to_table(c, u)


####################################################################
df2 = pr.read_excel('uloha5/uloha5.xlsx', 'List1', 'J2:O25')
df2.columns = ['r', 'dr', 'i', 'di', 'du', 'ddu']
r = pr.Var(df2['r'], df2['dr'], 'R_z', 'k\\Omega')
i = pr.Var(df2['i'], df2['di'], 'I_{SS}', 'mA')
u = pr.Var(df2['du'], df2['ddu'], '\\Delta U', 'V')

pr.to_table(r, i, u)

fig2, ax2 = plt.subplots(layout='constrained')
rel2 = pr.Rel(i, u, pr.F.linear)
rel2.fit()
rel2.plot_data(ax2)
ax2.legend()

plt.show()


