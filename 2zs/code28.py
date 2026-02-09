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

kToverq = 1.380649e-23 * (273.15+21) / 1.602176634e-19  # V at 21°C
print('kT/q:', kToverq)

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
figsun, axssun = plt.subplots(ncols=2, layout='constrained', figsize=(13,5))
axsunu = axssun[0]
axsuni = axsunu.twinx()
axsunu.scatter(sun.val, u.val, marker='^', 
                   label='napětí naprázdno', color='blue')
axsuni.scatter(sun.val, i.val, marker='s', 
                   label='zkratový proud', color='black')
splineu = make_interp_spline(sun.val, u.val, k=2)
splinei = make_interp_spline(sun.val, i.val, k=2)
xsun = np.linspace(sun.val.min(), sun.val.max(), 300)
axsunu.plot(xsun, splineu(xsun), color='blue', linestyle=':')
axsuni.plot(xsun, splinei(xsun), color='black', linestyle=':')
axsunu.set_xlabel(sun.long_name)
axsunu.set_ylabel(u.long_name)
#axsuni.set_xlabel(sun.long_name)
axsuni.set_ylabel(i.long_name)

lines, labels = [], []
for ax in [axsunu, axsuni]:
    line, label = ax.get_legend_handles_labels()
    lines += line
    labels += label
axsunu.legend(lines, labels, loc='upper left')

figsun.set_constrained_layout_pads(wspace=0.05)

figsun.savefig('uloha28/uisun.png', dpi=300)
plt.close(figsun)


print('Done table:')
pr.to_table(sun, u, i)

fig, axs = plt.subplots(ncols=2,layout='constrained', figsize=(10,5))
pr.plot(fig, axs[0], u, i, fit_curve=False)
axs[0].set_yscale('log')
#pr.plot(fig, axs[1], sun, u, fit_curve=False)
axs[1].semilogy(u.val, i.val, 'o', label='Data')
plt.close(fig)
#plt.show()
u2 = Var(u.val[-6:-1], errors=u.err[-6:-1], short_name='U_{OC}', unit='V')
i2 = Var(i.val[-6:-1], errors=i.err[-6:-1], short_name='I_{SC}', unit='mA')
logi2 = Var(np.log(i2.val), errors=i2.err / i2.val, short_name='ln(I)', unit='')
logi = Var(np.log(i.val), errors=i.err / i.val, short_name='ln(I)', unit='')
fig2, ax2 = plt.subplots(layout='constrained')
isc_rel = pr.Rel(u2, logi2, pr.F.linear)
isc_rel.fit(p0=[1e-2, 0.5])
#isc_rel.plot_data(ax2, err=(0,0))
#isc_rel.plot_fit(ax2)
x = np.linspace(min(u.val), max(u.val), 200)
axssun[1].plot(x, isc_rel.func(x, *isc_rel.coeffs.val), linestyle=':', color='black', linewidth=1.5, 
         label='lineární závislost pro největší osvětlení')
axssun[1].scatter(u.val, logi.val, marker='s', color='black', s=25, label='naměřené hodnoty')
axssun[1].set_xlabel(u.long_name)
axssun[1].set_ylabel('$\\ln(I_{SC}) \\, (\\ln(\\mathrm{mA}))$')
axssun[1].legend()
figsun.savefig('uloha28/sun.png', dpi=300)
#fig2.savefig('uloha28/ioveru.png', dpi=300)
plt.close(fig2)

print('i0, n', isc_rel.coeffs)
print(np.exp(isc_rel.coeffs.val[0]), 
      np.exp(isc_rel.coeffs.val[0]) * isc_rel.coeffs.err[0],
    isc_rel.coeffs.unc[1] * kToverq)

########################### fotoclanek se zatezi Rl ###########################
# Rlopt and RL ranges
df2 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='A18:F21')
df2.columns = ['psun', 'ouc', 'isc', 'rlopt', 'rmax', 'rmin']
psun = Var(df2['psun'].to_numpy(), errors=0.01, short_name='P/P_{Sun}')
print(u.err[[4,7,12]])
ouc = Var(df2['ouc'].to_numpy(), errors=u.err[[4,7,12]], short_name='U_{OC}', unit='V')
isc = Var(df2['isc'].to_numpy(), errors=i.err[[4,7,12]], short_name='I_{SC}', unit='mA')
rlopt = ouc / isc * 1000
rlopt.set_lname('R_{L,opt}', '\\Omega')
rmax = 10 * rlopt
rmax.set_lname('R_{L,max}', '\\Omega')
rmin = rlopt / 10
rmin.set_lname('R_{L,min}', '\\Omega')
print('Done table 2:')
pr.to_table(psun, ouc, isc, rlopt, rmin, rmax)

############################## 0.25 sun #############################
df3 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='A24:E46')
df3.columns = ['rl','u','i','P','du']
df3.sort_values(by='rl', inplace=True)
isc_25 = 22.212


u25 = Var(df3['u'].to_numpy(), errors=df3['du'].to_numpy(), short_name='U', unit='V')
rl25 = Var(df3['rl'].to_numpy(), errors='1%', short_name='R_L', unit='\\Omega')
i25 = u25 / rl25 * 1000
i25.set_lname('I', 'mA')
p25 = u25 * i25
p25.set_lname('P', 'mW')
print('Tabulka 25:')
pr.to_table(rl25, u25, i25, p25)

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

#i_rel_25.fit(p0=[isc_25, 0.005, n])
#i_rel_25.plot_data(axi_25, color='green', err=(0,0))
axi_25.plot(u25.val, i25.val, 's', color='green', linestyle=':', label='proud')
axp_25.plot(u25.val, p25.val, '^', color='red', linestyle=':', label='výkon')
axi_25.set_ylabel('I (mA)')
axp_25.set_ylabel('P (mW)')
axi_25.set_xlabel('U (V)')
#spline_25_i = make_interp_spline(u25.val, i25.val, k=2)
#x_25 = np.linspace(u25.val.min(), u25.val.max(), 300)
#axi_25.plot(x_25, spline_25_i(x_25), color='green', linestyle=':')
#i_rel_25.plot_fit(axi_25, color='green')
#p_rel_25.fit(p0=[isc_25, 0.005, n])
#p_rel_25.plot_data(axp_25, color='red', marker='^', err=(0,0))
#spline_25_p = make_interp_spline(u25.val, p25.val, k=2)
#axp_25.plot(x_25, spline_25_p(x_25), color='red', linestyle=':')
#p_rel_25.plot_fit(axp_25, color='red')
# combine legends
lines_25, labels_25 = [], []
for ax in [axi_25, axp_25]:
    line, label = ax.get_legend_handles_labels()
    lines_25 += line
    labels_25 += label
axi_25.legend(lines_25, labels_25, loc='upper right')

id_25 = np.argmax(p25.val)
pmpp25 = p25.unc[id_25] # mW
umpp25 = u25.unc[id_25] # V
impp25 = i25.unc[id_25] # mA
rmpp25 = umpp25 / impp25 * 1000 # Ohm
ff25 = pmpp25 / (isc.unc[0] * ouc.unc[0])
eta25 = pmpp25 / (psun.unc[0] * 100 * 3)

xmin, xmax = axi_25.get_xlim()
ymin, ymax = axp_25.get_ylim()
axi_25.axhline(y=impp25.nominal_value, xmax=(umpp25.nominal_value-xmin)/(xmax - xmin),
                color='black', linestyle=':')
axp_25.axvline(x=umpp25.nominal_value, ymax=(pmpp25.nominal_value-ymin)/(ymax - ymin),
               color='black', linestyle=':')
fig25.savefig('uloha28/sun25.png', dpi=300)
plt.close(fig25)

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

################################# 0.5 sun ####################################
df4 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='J3:N23')
isc_50 = 47.088
df4.columns = ['rl','u','i','P','du']
df4.sort_values(by='rl', inplace=True)

u50 = Var(df4['u'].to_numpy(), errors=df4['du'].to_numpy(), short_name='U', unit='V')
rl50 = Var(df4['rl'].to_numpy(), errors='1%', short_name='R_L', unit='\\Omega')
i50 = u50 / rl50 * 1000
i50.set_lname('I', 'mA')
p50 = u50 * i50
p50.set_lname('P', 'mW')

print('Tabulka 50:')
pr.to_table(rl50, u50, i50, p50)

i_rel_50 = pr.Rel(u50, i50, loaded_fotocell_i)
p_rel_50 = pr.Rel(u50, p50, loaded_fotocell_p)

axi_50.plot(u50.val, i50.val, 's', color='green', linestyle=':', label='proud')
axp_50.plot(u50.val, p50.val, '^', color='red', linestyle=':', label='výkon')
axi_50.set_ylabel('I (mA)')
axp_50.set_ylabel('P (mW)')
axi_50.set_xlabel('U (V)')

# i_rel_50.fit(p0=[isc_50, 0.01, n])
# i_rel_50.plot_data(axi_50, color='green', err=(0,0))
# i_rel_50.plot_fit(axi_50, color='green')
# p_rel_50.fit(p0=[isc_50, 0.01, n])
# p_rel_50.plot_data(axp_50, color='red', marker='^', err=(0,0))
# p_rel_50.plot_fit(axp_50, color='red')

lines_50, labels_50 = [], []
for ax in [axi_50, axp_50]:
    line, label = ax.get_legend_handles_labels()
    lines_50 += line
    labels_50 += label
axi_50.legend(lines_50, labels_50, loc='upper right')


id_50 = np.argmax(p50.val)
pmpp50 = p50.unc[id_50] # mW
print(pmpp50)
umpp50 = u50.unc[id_50] # V
impp50 = i50.unc[id_50] # mA
rmpp50 = umpp50 / impp50 * 1000 # Ohm
ff50 = pmpp50 / (isc.unc[1] * ouc.unc[1])
eta50 = pmpp50 / (psun.unc[1] * 100 * 3)

xmin, xmax = axi_50.get_xlim()
ymin, ymax = axp_50.get_ylim()
axi_50.axhline(y=impp50.nominal_value, xmax=(umpp50.nominal_value-xmin)/(xmax - xmin),
                color='black', linestyle=':')
axp_50.axvline(x=umpp50.nominal_value, ymax=(pmpp50.nominal_value-ymin)/(ymax - ymin),
               color='black', linestyle=':')
fig50.savefig('uloha28/sun50.png', dpi=300)
plt.close(fig50)


################################## 1.0 sun #############################
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

print('Tabulka 10:')
pr.to_table(rl10, u10, i10, p10)

i_rel_10 = pr.Rel(u10, i10, loaded_fotocell_i)
p_rel_10 = pr.Rel(u10, p10, loaded_fotocell_p)

axi_10.plot(u10.val, i10.val, 's', color='green', linestyle=':', label='proud')
axp_10.plot(u10.val, p10.val, '^', color='red', linestyle=':', label='výkon')
axi_10.set_ylabel('I (mA)')
axp_10.set_ylabel('P (mW)')
axi_10.set_xlabel('U (V)')

# i_rel_10.fit(p0=[isc_10, 0.02, n])
# i_rel_10.plot_data(axi_10, color='green', err=(0,0))
# i_rel_10.plot_fit(axi_10, color='green')
# p_rel_10.fit(p0=[isc_10, 0.02, n])
# p_rel_10.plot_data(axp_10, color='red', marker='^', err=(0,0))
# p_rel_10.plot_fit(axp_10, color='red')
# print(i_rel_10.coeffs)
# print(p_rel_10.coeffs)

lines_10, labels_10 = [], []
for ax in [axi_10, axp_10]:
    line, label = ax.get_legend_handles_labels()
    lines_10 += line
    labels_10 += label
axi_10.legend(lines_10, labels_10, loc='upper right')


id_10 = np.argmax(p10.val)
pmpp10 = p10.unc[id_10] # mW
umpp10 = u10.unc[id_10] # V
print(umpp10)
impp10 = i10.unc[id_10] # mA
rmpp10 = umpp10 / impp10 * 1000 # Ohm
ff10 = pmpp10 / (isc.unc[2] * ouc.unc[2])
eta10 = pmpp10 / (psun.unc[2] * 100 * 3)
print('xxxxxxxxxx limit', axi_10.get_xlim()[1])
xmin, xmax = axi_10.get_xlim()
ymin, ymax = axp_10.get_ylim()

axi_10.axhline(y=impp10.nominal_value, xmax=(umpp10.nominal_value-xmin)/(xmax - xmin),
                color='black', linestyle=':')
axp_10.axvline(x=umpp10.nominal_value, ymax=(pmpp10.nominal_value-ymin)/(ymax - ymin),
               color='black', linestyle=':')
fig10.savefig('uloha28/sun10.png', dpi=300)
plt.close(fig10)

uoc_50 = ouc.unc[1]
uoc_10 = ouc.unc[2]
isc_50 = isc.unc[1]
isc_10 = isc.unc[2]
delta = 30 # mA
u1 = 0.4265
u2 = 0.5017

figtwo, axtwo = plt.subplots(layout='constrained')
axtwo.plot(u10.val, i10.val, 's:', color='green', label='intenzita 1.0 Sun')
axtwo.plot(u50.val, i50.val, '^:', color='red', label='intenzita 0.5 Sun')
xmin, xmax = axtwo.get_xlim()
ymin, ymax = axtwo.get_ylim()
axtwo.set_ylabel(i50.long_name)
axtwo.set_xlabel(u50.long_name)
axtwo.legend()
axtwo.axhline(y=isc_50.nominal_value - delta, xmax=(u2-xmin)/(xmax - xmin), 
              color='black', linestyle=':')
axtwo.axhline(y=isc_10.nominal_value - delta, xmax=(u1-xmin)/(xmax - xmin),
              color='black', linestyle=':')
axtwo.axvline(x=u1, ymax=(isc_10.nominal_value-delta-ymin)/(ymax - ymin),
               color='black', linestyle=':')
axtwo.axvline(x=u2, ymax=(isc_50.nominal_value-delta-ymin)/(ymax - ymin),
               color='black', linestyle=':')
figtwo.savefig('uloha28/double.png', dpi=300)
#plt.close(figtwo)

pmaxs = Var(np.array([pmpp25.nominal_value, pmpp50.nominal_value, pmpp10.nominal_value]),
            errors=np.array([pmpp25.std_dev, pmpp50.std_dev, pmpp10.std_dev]),
            short_name='P_{mpp}', unit='mW')
umaxs = Var(np.array([umpp25.nominal_value, umpp50.nominal_value, umpp10.nominal_value]),
            errors=np.array([umpp25.std_dev, umpp50.std_dev, umpp10.std_dev]),
            short_name='U_{mpp}', unit='V')
imaxs = Var(np.array([impp25.nominal_value, impp50.nominal_value, impp10.nominal_value]),
            errors=np.array([impp25.std_dev, impp50.std_dev, impp10.std_dev]),
            short_name='I_{mpp}', unit='mA')
rmaxs = Var(np.array([rmpp25.nominal_value, rmpp50.nominal_value, rmpp10.nominal_value]),
            errors=np.array([rmpp25.std_dev, rmpp50.std_dev, rmpp10.std_dev]),
            short_name='R_{mpp}', unit='\\Omega')
rlopts = Var(np.array([rlopt.val[0], rlopt.val[1], rlopt.val[2]]),
              errors=np.array([rlopt.err[0], rlopt.err[1], rlopt.err[2]]),
              short_name='R_{L,opt}', unit='\\Omega')
ffs = Var(np.array([ff25.nominal_value, ff50.nominal_value, ff10.nominal_value]),
        errors=np.array([ff25.std_dev, ff50.std_dev, ff10.std_dev]),
        short_name='FF', unit=None)
etas = Var(np.array([eta25.nominal_value, eta50.nominal_value, eta10.nominal_value]),
          errors=np.array([eta25.std_dev, eta50.std_dev, eta10.std_dev]),
          short_name='\\eta', unit=None)
print('Done table 3:')
pr.to_table(pmaxs, umaxs, imaxs, rmaxs, rlopts, ffs, etas)

############################### low intensity ###############################
df6 = pr.read_excel('uloha28/uloha28.xlsx', sheet_name='List1', cells='F25:J32')
df6.columns = ['udiod', 'usc', 'isc', 'du', 'di']
ud = Var(df6['udiod'].to_numpy(), errors=0.1, short_name='U_{D}', unit='V')
usc = Var(df6['usc'].to_numpy(), errors=df6['du'].to_numpy(), short_name='U_{OC}', unit='mV')
isc = Var(df6['isc'].to_numpy(), errors=df6['di'].to_numpy(), short_name='I_{SC}', unit='\\mu A')

print('Table low intensity:')
pr.to_table(ud, usc, isc)

usc2 = Var(usc.val[:4], errors=usc.err[:4], short_name='U_{OC}', unit='mV')
isc2 = Var(isc.val[:4], errors=isc.err[:4], short_name='I_{SC}', unit='\\mu A')
figlow, axlow = plt.subplots(layout='constrained')
#lowrel = pr.Rel(usc2, isc2, pr.F.direct)
lowrel2 = pr.Rel(usc, isc, pr.F.direct)
#quadratic = pr.Rel(usc, isc, pr.F.polynomial)
#quadratic.fit(p0=[-8e-3, 9e-2, 2e-4, 4e-6])
#quadratic.fit(p0=[9e-2, 7e-2, 2e-7])
#quadratic.plot_data(axlow, color='green', err=(0,0), label='hodnoty pro celý rozsah', marker='o', zorder=20, scale=0.8)
#quadratic.plot_fit(axlow, label='kvadratická závislost', color='green', linestyle='--')
#lowfit = Polynomial.fit(usc.val, isc.val, 2, w=1/isc.err)
#print(lowfit.convert().coef)
#lowrel.fit(p0=[1e-2])
lowrel2.fit(p0=[1e-2])
lowrel2.plot_data(axlow, err=(0,0))
lowrel2.plot_fit(axlow, label='teoretická lineární závislost')
#lowrel.plot_data(axlow, err=(0,0), zorder=10, label='hodnoty pro nejnižší osvětlení', scale=2.2)
#lowrel.plot_fit(axlow, label='lineární závislost', linewidth=2)
axlow.legend()
# axlow.set_yscale('log')
# axlow.set_xscale('log')
figlow.savefig('uloha28/lowlog.png', dpi=300)
plt.close(figlow)

#print('lin', lowrel.coeffs)
#print('quad',quadratic.coeffs)
#print(quadratic.coeffs.unc[1] -  2.341728935549905e-10 / kToverq / 1.2465141700790494)
print(2.341728935549905e-10 / (kToverq * 1.2465141700790494)**2)
print('lin2', lowrel2.coeffs)
print(1 / lowrel2.coeffs)
plt.show()