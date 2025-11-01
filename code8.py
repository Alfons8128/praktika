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
f1 = Var(df['f1'].to_numpy(), errors=0.1, short_name='f_1', unit='kHz')
f2 = Var(df['f2'].to_numpy(), errors=0.1, short_name='f_2', unit='kHz')
c = Var(rdf['c'].to_numpy(), errors=0.25, short_name='C_N', unit='pF')
fr = (f1 + f2) / 2
fr.set_lname('f_r', 'kHz')
w = (2 * np.pi * fr) ** -2 * 1e8
w.set_lname('\\omega_r^{-2}', '10^{-14}\\, s^{-2}')

f12 = Var(df2['f1'].to_numpy(), errors=0.1, short_name='f_1', unit='kHz')
f22 = Var(df2['f2'].to_numpy(), errors=0.1, short_name='f_2', unit='kHz')
c2 = Var(rdf2['c'].to_numpy(), errors=0.25, short_name='C_N', unit='pF')
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
print('Fit parameters:', pr.ufmt(a, 'eP'), pr.ufmt(b, 'eP'), pr.ufmt(b/a, 'eP'))

fit2, cov2  = curve_fit(pr.F.linear, c2.val, w2.val, sigma=w2.err, absolute_sigma=True)
d_fit2 = np.sqrt(np.diag(cov2))
a2, b2 = unp.uarray([*fit2],[*d_fit2])
print('Fit parameters:', pr.ufmt(a2, 'eP'), pr.ufmt(b2, 'eP'), pr.ufmt(b2/a2, 'eP'))

fig, ax = plt.subplots()
ax.plot(c.val, pr.F.linear(c.val, *fit), 'g:', linewidth=1.5, label='fitovaná přímka')
errorbar = False
if errorbar:
    ax.errorbar(c.val, w.val, yerr=w.err, xerr=c.err, fmt='gs', linewidth=1, markersize=5, capsize=3, label='naměřené hodnoty')
else:
    ax.scatter(c.val, w.val, marker='s', s=25, color='green', linewidth=1, label='cívka A')

ax.plot(c2.val, pr.F.linear(c2.val, *fit2), ':', color='brown', linewidth=1.5, label='fitovaná přímka')
ax.text(0.05, 0.5, 'sem se to vykresli', transform=ax.transAxes)
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
#fig.savefig('uloha8/singleAandB.png', dpi=300)
plt.show()
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
f13 = Var(df3['f1'].to_numpy(), errors=0.1, short_name='f_1', unit='kHz')
f23 = Var(df3['f2'].to_numpy(), errors=0.1, short_name='f_2', unit='kHz')
c3 = Var(rdf3['c'].to_numpy(), errors=0.25, short_name='C_N', unit='pF')
fr3 = (f13 + f23) / 2
fr3.set_lname('f_r', 'kHz')
w3 = (2 * np.pi * fr3) ** -2 * 1e8
w3.set_lname('\\omega_r^{-2}', '10^{-14}\\, s^{-2}')

f14 = Var(df4['f1'].to_numpy(), errors=0.1, short_name='f_1', unit='kHz')
f24 = Var(df4['f2'].to_numpy(), errors=0.1, short_name='f_2', unit='kHz')
c4 = Var(rdf4['c'].to_numpy(), errors=0.25, short_name='C_N', unit='pF')
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
print('Fit parameters:', pr.ufmt(a3, 'eP'), pr.ufmt(b3, 'eP'), pr.ufmt(b3/a3, 'eP'))

fit4, cov4  = curve_fit(pr.F.linear, c4.val, w4.val, sigma=w4.err, absolute_sigma=True)
d_fit4 = np.sqrt(np.diag(cov4))
a4, b4 = unp.uarray([*fit4],[*d_fit4])
print('Fit parameters:', pr.ufmt(a4, 'eP'), pr.ufmt(b4, 'eP'), pr.ufmt(b4/a4, 'eP'))
l1 = a3 * 1e4
l2 = a4 * 1e4
m = (l1-l2) / 4
print('vzájemná indukčnost:', pr.ufmt(m, 'eP'))
ll1 = a * 1e4 + a2 * 1e4 + 2 * m
ll2 = a * 1e4 + a2 * 1e4 - 2 * m
print('l1, l2', pr.ufmt(ll1,'P'), pr.ufmt(ll2,'P'))

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
                         label=f'zapojení ABI: ${w3.short_name} = {a3.nominal_value:.3f} \\cdot {c3.short_name} + {b3.nominal_value:.2f}$')
combined_handle2 = Line2D([], [], color='brown', marker='o', linestyle=':', 
                          label=f'zapojení ABII: ${w4.short_name} = {a4.nominal_value:.3f} \\cdot {c4.short_name} + {b4.nominal_value:.2f}$')

ax4.legend(handles=[combined_handle, combined_handle2])
ax4.set_xlabel(c3.long_name)
ax4.set_ylabel(w3.long_name)
fig4.savefig('uloha8/doubleAandB.png', dpi=300)
plt.close('all')
#plt.show()

dff = pr.read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='S2:U31')
dff.columns = ['freq', 'flfr', 'z']
freqq = Var(dff['freq'].to_numpy(), errors=0.1, name='f', unit='kHz')
z = Var(dff['z'].to_numpy(), errors=0.5, name='z', unit=None)
x = freqq / fr3.unc[4]
x.set_lname('x')
zr = ufloat(max(z.val), 0.5)
y2 = z / zr
y2.set_lname('y^2')
for xs in x.unc:
    print(f'{xs:.1uP}')
print(min(x.val), max(x.val))

fit5, cov5  = curve_fit(pr.F.resonance, x.val, y2.val, sigma=y2.err, absolute_sigma=True, p0=[0.036])
d_fit5 = np.sqrt(np.diag(cov5))
print(d_fit5)
print(fit5, cov5)
d5, = unp.uarray([*fit5],[*d_fit5])
print(d5)
print('Fit parameters:', pr.ufmt(d5, 'P'))

fig5, ax5 = plt.subplots()
ax5.plot(x.val, pr.F.resonance(x.val, *fit5), ':', color='black', linewidth=1.5, label='teoretická závislost')
errorbar = True
if errorbar:
    ax5.errorbar(x.val, y2.val, yerr=y2.err, color='black',fmt='s', linewidth=1, markersize=3, capsize=3, label='naměřené hodnoty')
else:
    ax5.scatter(x.val, y2.val, marker='s', s=15, color='black', linewidth=1, label='poměr výchylek galvanometru')

ax5.plot([min(x.val),max(x.val)],[0.5,0.5], label='$y^2=0,5$', linewidth=0.5)
ax5.legend()
ax5.set_xlabel(x.long_name)
ax5.set_ylabel(y2.long_name)
fig5.savefig('uloha8/resonance.png', dpi=300)
#plt.show()
plt.close('all')

q = 1 / d5
rs = d5 * 2 * np.pi * (fr3.unc[4] * 1000) * (a3 / 100)
print(a3)
print('d:', pr.ufmt(d5,'P'))
print('q:', pr.ufmt(q,'P'))
print('rs:', pr.ufmt(rs,'P'))


print('All done!')