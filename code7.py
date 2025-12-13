import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

import praktika as pr

df1 = pr.read_excel(file_path='uloha7/uloha7.xlsx', sheet_name='List1', cells='A3:H8')
df1.columns = ['ua', 'dua', 'ia', 'dia', 'uc', 'duc', 'ic', 'dic']

f = 50  # Hz
f = pr.Var(f, 0.0001*f, 'f', 'Hz')
w = 2 * np.pi * f
w.set_lname('\\omega', 's^{-1}')

df2 = pr.read_excel(file_path='uloha7/uloha7.xlsx', sheet_name='List1', cells='A11:H16')
df2.columns = ['uf', 'duf', 'if', 'dif', 'ucf', 'ducf', 'icf', 'dicf']

ua = pr.Var(df1['ua'].to_numpy(), df1['dua'].to_numpy(), 'U_A', 'V')
ia = pr.Var(df1['ia'].to_numpy(), df1['dia'].to_numpy(), 'I_A', 'mA')
uc = pr.Var(df1['uc'].to_numpy(), df1['duc'].to_numpy(), 'U_C', 'V')
ic = pr.Var(df1['ic'].to_numpy(), df1['dic'].to_numpy(), 'I_C', 'mA')

uf = pr.Var(df2['uf'].to_numpy(), df2['duf'].to_numpy(), 'U_F', 'V')
if_ = pr.Var(df2['if'].to_numpy(), df2['dif'].to_numpy(), 'I_F', 'mA')
ucf = pr.Var(df2['ucf'].to_numpy(), df2['ducf'].to_numpy(), 'U_{CF}', 'V')
icf = pr.Var(df2['icf'].to_numpy(), df2['dicf'].to_numpy(), 'I_{CF}', 'mA')
#print(df1)
#print(df2)
#pr.to_table(ua, ia, uc, ic, uf, if_, ucf, icf)

rela = pr.Rel(ua, ia, pr.F.direct)
relc = pr.Rel(uc, ic, pr.F.direct)
relf = pr.Rel(uf, if_, pr.F.direct)
relfc = pr.Rel(ucf, icf, pr.F.direct)

rela.fit()
relc.fit()
relf.fit()
relfc.fit()

figc, axc = plt.subplots(layout='constrained')
rela.plot_data(axc, err=(1,0), label='kondenzátor A', color='green')
relc.plot_data(axc, err=(1,0), label='kondenzátor C', color='blue')
relf.plot_data(axc, err=(1,0), label='kondenzátor F', color='red')
relfc.plot_data(axc, err=(1,1), label='kondenzátory C a F', color='purple')

rela.plot_fit(axc, color='green')
relc.plot_fit(axc, color='blue')
relf.plot_fit(axc, color='red')
relfc.plot_fit(axc, color='purple')

ka = rela.coeffs[0]
kc = relc.coeffs[0]
kf = relf.coeffs[0]
kfc = relfc.coeffs[0]

#print(rela.coeffs, relc.coeffs, relf.coeffs, relfc.coeffs)
combined_handle1 = Line2D([], [], color='green', marker='s', linestyle=':', 
                         label=f'kondenzátor A: ${ia.short_name} = {ka.val[0]:.3f} \\cdot {ua.short_name}$')
combined_handle2 = Line2D([], [], color='blue', marker='s', linestyle=':', 
                          label=f'kondenzátor C: ${ic.short_name} = {kc.val[0]:.3f} \\cdot {uc.short_name}$')
combined_handle3 = Line2D([], [], color='red', marker='s', linestyle=':', 
                          label=f'kondenzátor F: ${if_.short_name} = {kf.val[0]:.3f} \\cdot {uf.short_name}$')
combined_handle4 = Line2D([], [], color='purple', marker='s', linestyle=':', 
                          label=f'kondenzátory C a F: ${icf.short_name} = {kfc.val[0]:.3f} \\cdot {ucf.short_name}$')

axc.legend(handles=[combined_handle1, combined_handle2, combined_handle3, combined_handle4])

axc.set_xlabel('U (V)')
axc.set_ylabel('I (mA)')

figc.savefig('uloha7/kond.png')
plt.close(figc)

ca = ka / (2 * np.pi * f) * 1000 # micro F
cc = kc / (2 * np.pi * f) * 1000
cf = kf / (2 * np.pi * f) * 1000
cfc = kfc / (2 * np.pi * f) * 1000
#print('Kapacity:', ca, cc, cf, cfc)
ks = pr.Var([ka.val[0], kc.val[0], kf.val[0], kfc.val[0]],
            [ka.err[0], kc.err[0], kf.err[0], kfc.err[0]],
            short_name='k', unit='mA/V')

caps = pr.Var([ca.val[0], cc.val[0], cf.val[0], cfc.val[0]],
               [ca.err[0], cc.err[0], cf.err[0], cfc.err[0]],
               short_name='C', unit='\\mu F')
#pr.to_table(ks, caps)

#####################################################




plt.show()

print('Done.')