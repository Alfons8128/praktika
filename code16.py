import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat, unumpy as unp
import praktika as pr
from matplotlib.lines import Line2D

df0 = pr.read_excel('uloha16/uloha16.ods', cells='A3:F4')
df = pr.read_excel('uloha16/uloha16.ods', cells='A6:G20')
df2 = pr.read_excel('uloha16/uloha16.ods', cells='I3:J17')

alpha1 = pr.Var(df0.iloc[0,0], df0.iloc[0,1], '\\alpha_1', '\\degree')
alpha2 = pr.Var(df0.iloc[0,2], df0.iloc[0,3], '\\alpha_2', '\\degree')
phi = 180 - alpha2 + alpha1
phi.set_lname('\\phi', '\\degree')
#print(phi)

col_names = df.columns.to_list()
colors = df['barva']

beta1 = pr.Var(df[f'{col_names[1]}'], df[f'{col_names[2]}'], '\\beta_1', '\\degree')
beta2 = pr.Var(df[f'{col_names[3]}'], df[f'{col_names[4]}'], '\\beta_2', '\\degree')
delta = (beta1 - beta2)/2
delta.set_lname('\\delta', '\\degree')
#delta_libre_office = pr.Var(df[f'{col_names[5]}'], df[f'{col_names[6]}'], '\\delta', '\\degree')
#print(delta)
#print(delta_libre_office)
lambda_tab = df2['lambda']
lambdas = pr.Var(lambda_tab, 0, '\\lambda', 'nm')

n = pr.sin((delta.radians() + phi.radians())/2) / pr.sin(phi.radians()/2)
n.set_lname('\\mathrm{N}')
print(n)
dispersion = pr.Rel(lambdas, n, pr.F.dispersion)

dispersion.fit(p0=[-160, 7.6, 1.5], get_unit=False)
fig, ax = plt.subplots()
dispersion.plot_data(ax, err=(0,0))
dispersion.plot_fit(ax, label='fitovaná disperzní křivka')
dispersion.show_equation(ax, combined=False)

eq_string = f'${dispersion.y.short_name} = {dispersion.coeffs[2].val[0]:.3f} + \
                    \\frac{{{dispersion.coeffs[1].val[0]:.1f}}} \
                    {{{dispersion.x.short_name} - {-dispersion.coeffs[0].val[0]:.0f}}}$'
eq_handle = Line2D([], [], color=dispersion.color, marker='', linestyle='', label=eq_string)
dispersion.fit_curve.set_label(f'{dispersion.fit_curve.get_label()}:')
ax.legend(handles=[dispersion.data_curve, dispersion.fit_curve, eq_handle])
#lambda0 = -160
#a = 7.6
#n0 = 1.5
#ax.plot(lambdas.val, pr.F.dispersion(lambdas.val, lambda0, a, n0))
fig.savefig('uloha16/dispersion.png', dpi=300)
plt.close(fig)
plt.show()

for c in dispersion.coeffs:
    print(c)

lambdaF = 486.1
lambdaC = 656.3
lambdaD = 589.3
nF = pr.F.dispersion(lambdaF, *dispersion.coeffs)
nC = pr.F.dispersion(lambdaC, *dispersion.coeffs)
nD = pr.F.dispersion(lambdaD, *dispersion.coeffs)
Delta = nF - nC
Delta.set_lname('\\Delta', '')
delta = Delta / (nD - 1)
delta.set_lname('\\delta', '')
Abbe = 1 / delta
Abbe.set_lname('\\gamma', '')
print(f'nD = {nD}')
print(f'Delta = nF - nC = {Delta}')
print(f'delta = (nF - nC) / (nD - 1) = {delta}')
print(f'Abbe number = (nD - 1) / (nF - nC) = {Abbe}')
pr.best_print(Delta)
pr.best_print(delta)
pr.best_print(Abbe)

print('Done code16.')