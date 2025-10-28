import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import praktika as pr
from praktika import Var
from matplotlib.lines import Line2D

############################ Isc a Uoc na intenzite ############################
rdf = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='A2:E15')
rdf.columns = ['sun','u','i','du','di']
rdf.sort_values(by='sun', inplace=True)

sun = Var(rdf['sun'].to_numpy(), errors=0.01, short_name='P/P_{Sun}')
u = Var(rdf['u'].to_numpy(), errors=rdf['du'].to_numpy(), short_name='U', unit='V')
i = Var(rdf['i'].to_numpy(), errors=rdf['di'].to_numpy(), short_name='I', unit='mA')

print('Done table:')
pr.to_table(sun, u, i)

fig, axs = plt.subplots(ncols=2,layout='constrained', figsize=(10,5))
pr.plot(fig, axs[0], u, i, fit_curve=False)
axs[0].set_yscale('log')
#pr.plot(fig, axs[1], sun, u, fit_curve=False)
axs[1].semilogy(u.val, i.val, 'o', label='Data')
plt.show()

# Rlopt and RL ranges
df2 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='A18:F21')

# 0.25 sun
df3 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='A24:E46')

# 0.5 sun
df4 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='J3:N23')

# 0.1 sun
df5 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='O3:S21')

# low intensity
df6 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='F25:J32')