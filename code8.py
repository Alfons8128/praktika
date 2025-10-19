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
w.set_lname('\\omega_r^{-2}', '10^{-14}\\, s^{-2}')

f12 = Var(df2['f1'].to_numpy(), errors=0.1, name='f_1', unit='kHz')
f22 = Var(df2['f2'].to_numpy(), errors=0.1, name='f_2', unit='kHz')
c2 = Var(rdf2['c'].to_numpy(), errors=0.25, name='C_N', unit='pF')
fr2 = (f12 + f22) / 2
fr2.set_lname('f_r', 'kHz')
w2 = (2 * np.pi * fr2) ** -2 * 1e8
w2.set_lname('\\omega_r^{-2}', '10^{-14}\\, s^{-2}')

print('Computed fr:', fr)
print('Computed w:', w)
for wr in w.unc:
    print(f'{wr:.2uP}')
print('Done table', pr.to_table(c, f1, f2, fr, w))
for wr in w2.unc:
    print(f'{wr:.2uP}')
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

combined_handle = Line2D([], [], color='green', marker='s', linestyle=':', 
                         label=f'cívka A: ${w.name} = {a.nominal_value:.3f} \\cdot {c.name} + {b.nominal_value:.2f}$')
combined_handle2 = Line2D([], [], color='brown', marker='o', linestyle=':', 
                          label=f'cívka B: ${w2.name} = {a2.nominal_value:.3f} \\cdot {c2.name} + {b2.nominal_value:.2f}$')

ax.legend(handles=[combined_handle, combined_handle2])
ax.set_xlabel(c.long_name)
ax.set_ylabel(w.long_name)
fig.savefig('uloha8/singleAandB.png', dpi=300)
#plt.show()
#plt.close('all')

##############################
rdf3 = pr.read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='K26:O31')
rdf4 = pr.read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='K3:O8')
rdf3.columns = ['fr','f1','f2','c','w']
rdf4.columns = ['fr','f1','f2','c','w']

df3 = pd.DataFrame()
df4 = pd.DataFrame()
for i in range(len(rdf3['f1'])):
    df3.loc[i, 'f1'] = min(rdf3.loc[i, 'f1'], rdf3.loc[i, 'f2'])
    df3.loc[i, 'f2'] = max(rdf3.loc[i, 'f1'], rdf3.loc[i, 'f2'])

for i in range(len(rdf4['f1'])):
    df4.loc[i, 'f1'] = min(rdf4.loc[i, 'f1'], rdf4.loc[i, 'f2'])
    df4.loc[i, 'f2'] = max(rdf4.loc[i, 'f1'], rdf4.loc[i, 'f2'])

print(df3)
f13 = Var(df3['f1'].to_numpy(), errors=0.1, name='f_1', unit='kHz')
f23 = Var(df3['f2'].to_numpy(), errors=0.1, name='f_2', unit='kHz')
c3 = Var(rdf3['c'].to_numpy(), errors=0.25, name='C_N', unit='pF')
fr3 = (f13 + f23) / 2
fr3.set_lname('f_r', 'kHz')
w3 = (2 * np.pi * fr3) ** -2 * 1e8
w3.set_lname('\\omega_r^{-2}', '10^{-14}\\, s^{-2}')

f14 = Var(df4['f1'].to_numpy(), errors=0.1, name='f_1', unit='kHz')
f24 = Var(df4['f2'].to_numpy(), errors=0.1, name='f_2', unit='kHz')
c4 = Var(rdf4['c'].to_numpy(), errors=0.25, name='C_N', unit='pF')
fr4 = (f14 + f24) / 2
fr4.set_lname('f_r', 'kHz')
w4 = (2 * np.pi * fr4) ** -2 * 1e8
w4.set_lname('\\omega_r^{-2}', '10^{-14}\\, s^{-2}')

print('Computed fr:', fr3)
print('Computed w:', w3)
for wr in w3.unc:
    print(f'{wr:.2uP}')
print('Done table', pr.to_table(c3, f13, f23, fr3, w3))
for wr in w4.unc:
    print(f'{wr:.2uP}')
print('Done table2', pr.to_table(c4, f14, f24, fr4, w4))

fit3, cov3  = curve_fit(pr.F.linear, c3.val, w3.val, sigma=w3.err, absolute_sigma=True)
d_fit3 = np.sqrt(np.diag(cov3))
a3, b3 = unp.uarray([*fit3],[*d_fit3])
print('Fit parameters:', pr.ufmt(a3, 'eL'), pr.ufmt(b3, 'eL'))

fit4, cov4  = curve_fit(pr.F.linear, c4.val, w4.val, sigma=w4.err, absolute_sigma=True)
d_fit4 = np.sqrt(np.diag(cov4))
a4, b4 = unp.uarray([*fit4],[*d_fit4])
print('Fit parameters:', pr.ufmt(a4, 'eL'), pr.ufmt(b4, 'eL'))

fig4, ax4 = plt.subplots()
ax4.plot(c3.val, pr.F.linear(c3.val, *fit3), 'g:', linewidth=1.5, label='fitovaná přímka')
errorbar = False
if errorbar:
    ax4.errorbar(c3.val, w3.val, yerr=w3.err, xerr=c3.err, fmt='gs', linewidth=1, markersize=5, capsize=3, label='naměřené hodnoty')
else:
    ax4.scatter(c3.val, w3.val, marker='s', s=25, color='green', linewidth=1, label='cívky ABI')

ax4.plot(c4.val, pr.F.linear(c4.val, *fit4), ':', color='brown', linewidth=1.5, label='fitovaná přímka')
errorbar = False
if errorbar:
    ax4.errorbar(c4.val, w4.val, yerr=w4.err, xerr=c4.err, color='brown',fmt='o', linewidth=1, markersize=5, capsize=3, label='naměřené hodnoty')
else:
    ax4.scatter(c4.val, w4.val, marker='o', s=25, color='brown', linewidth=1, label='cívky ABII')

combined_handle = Line2D([], [], color='green', marker='s', linestyle=':', 
                         label=f'zapojení ABI: ${w3.name} = {a3.nominal_value:.3f} \\cdot {c3.name} + {b3.nominal_value:.2f}$')
combined_handle2 = Line2D([], [], color='brown', marker='o', linestyle=':', 
                          label=f'zapojení ABII: ${w4.name} = {a4.nominal_value:.3f} \\cdot {c4.name} + {b4.nominal_value:.2f}$')

ax4.legend(handles=[combined_handle, combined_handle2])
ax4.set_xlabel(c3.long_name)
ax4.set_ylabel(w3.long_name)
fig4.savefig('uloha8/doubleAandB.png', dpi=300)
#plt.show()



print('All done!')