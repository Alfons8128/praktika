import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat, unumpy as unp
import praktika as pr

#print(pr.excel_to_latex_2('uloha3/uloha3.ods', cells='B3:K7', format=None))
a = pr.Var(1654.67522452891,0.257575982412746,'a','nm')

df = pr.read_excel('uloha3/uloha3.ods',sheet_name='List1', cells='B10:G25')
col_names = df.columns.to_list()
colors = df['barva']

psi1 = pr.Var(df[f'{col_names[1]}'], 0.05, '\\psi_1', '\\degree')
psi2 = pr.Var(df[f'{col_names[2]}'], 0.05, '\\psi_2', '\\degree')
phi = (psi1 - psi2 + 360)/2
phi.set_lname('\\varphi_1', '\\degree')
lambda_tab = pr.Var(df[f'{col_names[5]}'], '1%', '\\lambda_{tab}', 'nm')
if isinstance(df[f'{col_names[5]}'][6], int):
    print('jsem int')
if isinstance(df[f'{col_names[5]}'][6], float):
    print('jsem float')

lambdas = a * pr.sin(phi/180*np.pi)

df2 = pr.read_excel('uloha3/uloha3.ods',sheet_name='List1', cells='K22:L36')
tabulkova = pr.Var(df2['tab'],'1%', '\\lambda_{tab}', 'nm')
kalib_err = np.zeros_like(tabulkova.err)
kalib_err[:7] = phi.err[:7]
kalib_err[7:] = phi.err[8:]
phi_kalib = pr.Var(df2['phi'], kalib_err, phi.short_name, phi.unit)

kalib = pr.Rel(phi_kalib, tabulkova, pr.F.quadratic)
#kalib2 = pr.Rel(phi, lambdas, pr.F.quadratic)

fig, ax = plt.subplots(layout='constrained')
kalib.fit(p0=[16,27,-0.1])
#print('phi', phi)
#print('lambdas', lambdas)
params, pcov = np.polynomial.Polynomial.fit(phi.val, lambdas.val, 2, full=True, w=1/phi.err**2)
#print('params:', params.convert().coef)
#print('pcov:', np.sqrt(pcov[2]))
kalib.coeffs = [pr.Var(param, np.sqrt(pcov[2][i]), f'c_{i}', '') for i, param in enumerate(params.convert().coef)]

kalib2 = pr.Rel(phi, lambdas, pr.F.quadratic)
kalib2.set_degree()
kalib2.fit()
for i,coeff in enumerate(kalib.coeffs):
    print('kalib:', coeff)
    print('kalib2:', kalib2.coeffs[i])

kalib.plot_data(ax, (0,0), label='tabelované vlnové délky')
kalib.plot_fit(ax, label='kalibrační křivka')
#print('handles', kalib.data_curve, kalib.fit_curve)

kalib.show_equation(ax, '.1f', combined=False)

#ax.legend()

fig.savefig('uloha3/calibration.png', dpi=300)
plt.close(fig)

calibred = phi * kalib.coeffs[1].val + kalib.coeffs[0].val + phi**2 * kalib.coeffs[2].val #+ phi**3 * kalib.coeffs[3].val + phi**4 * kalib.coeffs[4].val
#calibred2 = kalib2.coeffs[0] + phi * kalib2.coeffs[1] + phi**2 * kalib2.coeffs[2]
#print('calibred:', calibred)
residuals = lambdas - calibred
#print('residuals:', residuals)
#residuals2 = lambdas - calibred2
residuals.set_lname('\\lambda_{res}', 'nm')

resid = pr.Rel(phi, residuals, pr.F.quadratic)
resid.fit(p0=[0,0,0.01])
params, pcov = np.polynomial.Polynomial.fit(phi.val, residuals.val, 2, full=True, w=1/phi.err**2)
#print('params:', params.convert().coef)
#print('pcov:', np.sqrt(pcov[2]))
resid.coeffs = [pr.Var(param, np.sqrt(pcov[2][i]), f'c_{i}', '') for i, param in enumerate(params.convert().coef)]

for coeff in resid.coeffs:
    print('resid', coeff)
fig2, ax2 = plt.subplots(layout='constrained')
resid.plot_data(ax2, err=(0,1), label='rezidua oproti kalibrační křivce')
resid.plot_fit(ax2, label='fitovaná křivka')
ax2.legend()
#print('handles', resid.data_curve, resid.fit_curve)
#resid.show_equation(ax2, '.1f', combined=False)


fig2.savefig('uloha3/residuals.png', dpi=300)
plt.close(fig2)

plt.show()
rozlis1 = 589.0 / pr.Var(0.6, '10%', '\\delta \\lambda_1', 'nm')
rozlis2 = 589.0 / pr.Var(0.3, '10%', '\\delta \\lambda_2', 'nm')
#print('rozliseni experiment:', rozlis1, rozlis2)
D = 18 * 1e6 #nm
max_rozlis1 = 1 * 0.82 * D / a
max_rozlis2 = 2 * 0.82 * D / a
#print('rozliseni teoret:', max_rozlis1, max_rozlis2)


df3 = pd.DataFrame({
    'barva': colors,
    'psi1': psi1,
    'psi2': psi2,
    'phi': phi,
    'experimentální': lambdas,
    'tabulková': tabulkova
})
#print(df3.to_latex(index=False))
colors = pr.Var(np.ones_like(colors), 0, 'barva', '')
#print(colors, psi1, psi2, phi, lambdas, tabulkova)
#print(pr.to_table(colors, psi1, psi2, phi, lambdas, lambda_tab))

df4 = pr.read_excel('uloha3/uloha3.ods',sheet_name='List1', cells='B3:I7')
phi1l = pr.Var(df4.iloc[0,6], df4.iloc[0,7], '\\varphi_{1l}', '\\degree')*np.pi/180
phi1h = pr.Var(df4.iloc[1,6], df4.iloc[1,7], '\\varphi_{1h}', '\\degree')*np.pi/180
phi2l = pr.Var(df4.iloc[2,6], df4.iloc[2,7], '\\varphi_{2l}', '\\degree')*np.pi/180
phi2h = pr.Var(df4.iloc[3,6], df4.iloc[3,7], '\\varphi_{2h}', '\\degree')*np.pi/180
delta_phi1 = phi1h - phi1l
delta_phi2 = phi2h - phi2l
print('delta_phi1:', delta_phi1, 'delta_phi2:', delta_phi2)
print('phi1l:', phi1l, 'phi1h:', phi1h, 'phi2l:', phi2l, 'phi2h:', phi2h)
dl = 0.6*1e-6
dispNa1e = delta_phi1 / dl
dispNa2e = delta_phi2 / dl
print('dispersion Na experiment 1 rad:', dispNa1e)
print('dispersion Na experiment 2 rad:', dispNa2e)
dispNa1t = 1 / (a * pr.cos((phi1h + phi1l)/2)) * 1e6
dispNa2t = 1 / (a * pr.cos((phi2h + phi2l)/2)) * 1e6
print('dispersion Na theoretical 1 rad:', dispNa1t)
print('dispersion Na theoretical 2 rad:', dispNa2t)

phiHgl = pr.Var(phi.val[8], phi.err[8], '\\varphi_1', '\\degree')*np.pi/180
phiHgh = pr.Var(phi.val[9], phi.err[9], '\\varphi_2', '\\degree')*np.pi/180
lambdal = pr.Var(lambdas.val[8], lambdas.err[8], '\\lambda', 'nm')/1e6
lambdah = pr.Var(lambdas.val[9], lambdas.err[9], '\\lambda', 'nm')/1e6
delta_phiH = phiHgh - phiHgl
delta_lambdaH = lambdah - lambdal
dispHgl = delta_phiH / delta_lambdaH
print('dispersion Hg experiment rad:', dispHgl)
dispHgt = 1 / (a * pr.cos((phiHgh + phiHgl)/2)) * 1e6
print('dispersion Hg theoretical rad:', dispHgt)
print(dispNa1e * (phi1h - phi1l).err / (phi1h - phi1l).val)

print("Done.")