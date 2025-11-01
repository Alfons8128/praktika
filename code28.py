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

kToverq = 0.026

def solar_exp(x, i0, n, rp):
    return i0 * (np.exp(x / n / kToverq) - 1) + x / rp

def loaded_fotocell_i(x, isc, i0, n):
    return isc - i0 * (np.exp(x / 0.026 / n) - 1)

def loaded_fotocell_p(x, isc, i0, n):
    return x * (isc - i0 * (np.exp(x / 0.026 / n) - 1))
n = 2.5
iii000 = 22.212 / (np.exp(0.519 / (n * kToverq)) - 1)

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
plt.close(fig)
#plt.show()

########################### fotoclanek se zatezi Rl ###########################
# Rlopt and RL ranges
df2 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='A18:F21')

# 0.25 sun
df3 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='A24:E46')
df3.columns = ['rl','u','i','P','du']
df3.sort_values(by='rl', inplace=True)
isc_25 = 22.212


u25 = Var(df3['u'].to_numpy(), errors=df3['du'].to_numpy(), short_name='U_{OC}', unit='V')
rl25 = Var(df3['rl'].to_numpy(), errors='1%', short_name='R_L', unit='\\Omega')
i25 = u25 / rl25 * 1000
i25.set_lname('I_{SC}', 'mA')
p25 = u25 * i25
p25.set_lname('P', 'mW')

fig25, ax25 = plt.subplots(layout='constrained')
fig50, ax50 = plt.subplots(layout='constrained')
fig10, ax10 = plt.subplots(layout='constrained')
axi_25 = ax25
axp_25 = ax25.twinx()
axi_50 = ax50
axp_50 = ax50.twinx()
axi_10 = ax10
axp_10 = ax10.twinx()
i_rel_25 = pr.Rel(u25, i25, loaded_fotocell_i)
p_rel_25 = pr.Rel(u25, p25, loaded_fotocell_p)

i_rel_25.fit(p0=[0.005, isc_25, n])
i_rel_25.plot_data(axi_25, color='green', err=(0,0))
i_rel_25.plot_fit(axi_25, color='green')
p_rel_25.fit(p0=[0.005, isc_25, n])
p_rel_25.plot_data(axp_25, color='red', marker='^', err=(0,0))
p_rel_25.plot_fit(axp_25, color='red')

# isc = 22.2126  # mA
# x = np.linspace(0, 0.5, 100)
# y = loaded_fotocell(x, iii000, n)
# ax2.plot(x, y, label='Theoretical curve')
# ax2.set_yscale('log')
# ax2.legend()

# fit, mat = curve_fit(loaded_fotocell, u25.val, i25.val, p0=[iii000, n])

# y33 = loaded_fotocell(u25.val, *fit)
# print('Fitted parameters (I0, n):', fit)
# ax2.plot(u25.val, y33, label='Fitted curve', linestyle='--')
# ax2.legend()
# x = np.linspace(-0.2, 0.5, 100)
# y = solar_exp(x, 0.1, 1.2, 100)

# # Example data (x must be sorted for spline interpolation)
# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([0, 2, 1, 3, 2, 5])

# # Create a smooth spline that goes through all points
# spline = make_interp_spline(u25.val, i25.val, k=4)  # k=3 means cubic spline

# # Generate new x values for a smooth curve
# x_smooth = np.linspace(u25.val.min(), u25.val.max(), 300)
# y_smooth = spline(x_smooth)
# fig3, ax3 = plt.subplots()
# # Plot original points and the smooth curve
# ax3.scatter(u25.val, i25.val, marker='s', c='black', label='Data Points')
# ax3.plot(x_smooth, y_smooth, color='black', label='Smooth Curve')
# ax3.legend()
# plt.close(fig3)

# 0.5 sun
df4 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='J3:N23')
isc_50 = 47.088
df4.columns = ['rl','u','i','P','du']
df4.sort_values(by='rl', inplace=True)

u50 = Var(df4['u'].to_numpy(), errors=df4['du'].to_numpy(), short_name='U_{OC}', unit='V')
rl50 = Var(df4['rl'].to_numpy(), errors='1%', short_name='R_L', unit='\\Omega')
i50 = u50 / rl50 * 1000
i50.set_lname('I_{SC}', 'mA')
p50 = u50 * i50
p50.set_lname('P', 'mW')

i_rel_50 = pr.Rel(u50, i50, loaded_fotocell_i)
p_rel_50 = pr.Rel(u50, p50, loaded_fotocell_p)

i_rel_50.fit(p0=[0.01, isc_50, n])
i_rel_50.plot_data(axi_50, color='green', err=(0,0))
i_rel_50.plot_fit(axi_50, color='green')
p_rel_50.fit(p0=[0.01, isc_50, n])
p_rel_50.plot_data(axp_50, color='red', marker='^', err=(0,0))
p_rel_50.plot_fit(axp_50, color='red')

# 1.0 sun
df5 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='O3:S21')
isc_10 = 92.483
df5.columns = ['rl','u','i','P','du']
df5.sort_values(by='rl', inplace=True)

u10 = Var(df5['u'].to_numpy(), errors=df5['du'].to_numpy(), short_name='U_{OC}', unit='V')
rl10 = Var(df5['rl'].to_numpy(), errors='1%', short_name='R_L', unit='\\Omega')
i10 = u10 / rl10 * 1000
i10.set_lname('I_{SC}', 'mA')
p10 = u10 * i10
p10.set_lname('P', 'mW')

i_rel_10 = pr.Rel(u10, i10, loaded_fotocell_i)
p_rel_10 = pr.Rel(u10, p10, loaded_fotocell_p)

i_rel_10.fit(p0=[0.02, isc_10, n])
i_rel_10.plot_data(axi_10, color='green', err=(0,0))
i_rel_10.plot_fit(axi_10, color='green')
p_rel_10.fit(p0=[0.02, isc_10, n])
p_rel_10.plot_data(axp_10, color='red', marker='^', err=(0,0))
p_rel_10.plot_fit(axp_10, color='red')

# low intensity
df6 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='F25:J32')

plt.show()