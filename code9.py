import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import praktika as pr
from praktika import Var
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

########################### staticka va charakteristika ##########################
va = pd.read_csv('uloha9/nemec_VA.txt', sep='\\s+', header=None)
va.columns = ['i', 'iunit', 'itype', 'u', 'uunit', 'utype']
print(va)
i = pr.Var(va['i'], errors='1%', unit='mA', short_name='I')
u = pr.Var(va['u'], errors='1%', unit='V', short_name='U')
for val in i.unc:
    if val.nominal_value < 2:
        val.std_dev = 0.003 * val.nominal_value + 3*1e-4
        continue
    if val.nominal_value < 20:
        val.std_dev = 0.003 * val.nominal_value + 3*1e-3
        continue
    if val.nominal_value < 200:
        val.std_dev = 0.003 * val.nominal_value + 3*1e-2
        continue

for val in u.unc:
    if val.nominal_value < 0.2:
        val.std_dev = 0.005 * val.nominal_value + 3*1e-5
        continue
    if val.nominal_value < 2:
        val.std_dev = 0.005 * val.nominal_value + 3*1e-4
        continue

print('Table VA:')
pr.to_table(i, u)
rel = pr.Rel(i, u)
fig, ax = plt.subplots(layout='constrained')
rel.plot_data(ax, err=(0,0), label='napětí na termistoru')

idxmax = np.argmax(u.val)
xmin, xmax = ax.get_xlim()
ax.axhline(u.val[idxmax], xmax=(i.val[idxmax]-xmin) / (xmax-xmin), color='green', linewidth=1)
ax.scatter(i.val[idxmax], u.val[idxmax], color='green', marker='s',label='maximální napětí $U_{m}$')
#ax.plot(i.val, u.val, color='black', label='napětí na termistoru')
ax.legend()
fig.savefig('uloha9/va.png', dpi=300)
#plt.close(fig)

########################## odpor na teplote #########################

rt = pd.read_csv('uloha9/nemec_RT.txt', sep='\\s+', header=None)
rt.columns = ['pt', 'ptunit', 'pttype', 'rt', 'rtunit', 'rttype']
print(rt)
pt = pr.Var(rt['pt'].to_numpy()[2:], errors='1%', unit='\\Omega', short_name='R_{Pt,100}')
rt = pr.Var(rt['rt'].to_numpy()[2:], errors='1%', unit='k\\Omega', short_name='R_{term}')

for val in pt.unc:
    val.std_dev = 0.002 * val.nominal_value + 5*1e-5

for val in rt.unc:
    if val.nominal_value < 2:
        val.std_dev = 0.0015 * val.nominal_value + 3*1e-7
        continue
    if val.nominal_value < 20:
        val.std_dev = 0.0015 * val.nominal_value + 3*1e-6
        continue
    if val.nominal_value < 200:
        val.std_dev = 0.0015 * val.nominal_value + 3*1e-5
        continue

r0 = pr.Var(100, errors=0, short_name='R_{0,Pt100}', unit='\\Omega')
alpha = pr.Var(0.00385, errors=0, short_name='\\alpha', unit='K^{-1}')
t = (pt - r0) / (r0 * alpha)
t.set_lname('T', '°C')
temp = t + 273.15
temp.set_lname('T', 'K')


lnr = (1000 * rt).ln() # ln (r in Ohm)
lnr.set_lname('\\ln R', '\\ln \\Omega')

oneovert = 1000 / temp
oneovert.set_lname('T^{-1}', '(mK)^{-1}')

print('Table RT:')
pr.to_table(pt, temp, oneovert, rt, lnr)

rel2 = pr.Rel(oneovert, lnr, pr.F.linear)
fig2, ax2 = plt.subplots(layout = 'constrained')
rel2.plot_data(ax2, err=(0,0))
rel2.fit()
rel2.plot_fit(ax2, label='fitovaná lineární závislost $\\ln R = 0.04826 + 2467.2 \cdot \\frac{1}{T}$')
ax2.legend()
fig2.savefig('uloha9/rt.png', dpi=300)
#plt.close(fig2)

print('ln(rinf), b, rinf', rel2.coeffs)
lnrinf = Var(rel2.coeffs.val[0], rel2.coeffs.err[0])
rinf = lnrinf.exp()
print(rinf)
print(f'{unp.exp(rel2.coeffs.unc[0]):.L}')

rel3 = pr.Rel(temp, rt)
fig3, ax3 = plt.subplots(layout='constrained')
rel3.plot_data(ax3, err=(0,0), label='změřený odpor termistoru')
ax3.legend()
fig3.savefig('uloha9/rtlog.png', dpi=300)

kb = 1.380649 # e-23 J/K
na = 6.02214076 # e23 1/mol
R = kb * na  # J/(K mol)
b = pr.Var(rel2.coeffs.val[1], rel2.coeffs.err[1], short_name='b', unit='K')
b = b * 1000
du = b * 2 * R # J/mol
print('b:', b)
print('du:', du)
t0 = pr.Var(22.0, errors=0.4, unit='°C', short_name='T_{0}')
t0 = t0 + 273.15
tm = b / 2 * (1 - unp.sqrt(1 - 4 * t0.unc / b.unc))
print('tm:', tm)
print(b / unp.log(u.val[idxmax] / i.val[idxmax] / rinf.val[0] * 1000))
print(b, u.unc[idxmax], i.unc[idxmax], rinf.unc[0])

#plt.show()

