import numpy as np
import matplotlib.pyplot as plt
import praktika as pr
from uncertainties import umath as um
from scipy.stats import norm
from scipy.optimize import curve_fit

filename = 'uloha18/uloha18.ods'

##################################### kalibrace z geometrie ##############
dfgeometry = pr.read_excel(filename, cells='A2:B9')
geocols = dfgeometry.columns.tolist()

sigma_d = 0.2
sigma_delta_d = 0.05
d = pr.Var(dfgeometry[geocols[0]], sigma_d, 'd', unit='cm')
delta_d = pr.Var(dfgeometry[geocols[1]], sigma_delta_d, '\\Delta d', unit='cm')
rel_geo = pr.Rel(d, delta_d, pr.F.linear)
rel_geo.fit(absolute_sigma=True)

figgeo, axgeo = plt.subplots()
rel_geo.plot_data(axgeo,err=(0,1))
rel_geo.plot_fit(axgeo)
rel_geo.show_equation(axgeo, format=['.2f', '.4f'])
rel_geo.make_legend(axgeo)
figgeo.savefig('uloha18/geometrie.png', dpi=300)
plt.close(figgeo)
plt.show()
# for coeff in rel_geo.coeffs:
#     pr.best_print(coeff)
#print(rel_geo.chi)
theta_ufloat = 2*um.atan2(rel_geo.coeffs[1].unc[0],2)
theta = pr.Var(theta_ufloat.nominal_value, theta_ufloat.std_dev, '\\theta', unit='rad')
# pr.best_print(theta)
lambda_hene = 632.8e-3
d_f_geo = lambda_hene/ (2*pr.sin(theta/2))
d_f_geo.set_lname('d_{F,g}', unit='\\mu m')
# pr.best_print(d_f_geo)

#print(pr.to_table_2(d, delta_d))


############################### kalibrace z projekce ##########
dfprojekce = pr.read_excel(filename, cells='A12:E19')
projcols = dfprojekce.columns.tolist()
sigma_x = 0.005 # mm
N_proj = pr.NonErrorVar(dfprojekce[projcols[0]], 'N', fmt='.0f')
x1 = pr.Var(dfprojekce[projcols[1]], sigma_x, 'x_1', unit='mm')
x2 = pr.Var(dfprojekce[projcols[2]], sigma_x, 'x_2', unit='mm')
x = x1 - x2
x.set_lname('x', unit='mm')
d_f_proj = x / N_proj * 1000
d_f_proj.set_lname('d_{F,p}', unit='\\mu m')
#pr.best_print(d_f_proj)
d_f_proj_mean = pr.mean(d_f_proj)
# pr.best_print(d_f_proj_mean)

#print(pr.to_table_2(N_proj, x1, x2, x, d_f_proj))

############################## skutecne mereni castic ##############
d_f = d_f_proj_mean
sigma_f = 0.1 # ms

dfcastice = pr.read_excel(filename, cells='A23:E103')
casticecols = dfcastice.columns.tolist()
index = pr.NonErrorVar(dfcastice[casticecols[0]], 'částice', fmt='.0f')
n = pr.NonErrorVar(dfcastice[casticecols[1]], 'N', fmt='.0f')
t = pr.Var(dfcastice[casticecols[2]], sigma_f, 't', unit='ms')
f = n / t * 1000
f.set_lname('f', unit='Hz')
vx = f * d_f / 1000
vx.set_lname('v_x', unit='mm/s')

index1 = index[:40]
n1 = n[:40]

t1 = t[:40]
f1 = f[:40]
vx1 = vx[:40]
index2 = index[40:]
n2 = n[40:]

t2 = t[40:]
f2 = f[40:]
vx2 = vx[40:]
#print(len(f1), len(f2))
#print(pr.to_table_2(index1, n1, t1, f1, vx1, index2, n2, t2, f2, vx2))

############################## analysis
vmin = min(vx)
vmax = max(vx)
delta_v = (vmax - vmin) / 9
# print(vmin, vmax)
print(delta_v)

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
bins1 = np.arange(1.5, 15.1, 1.5)
bins2 = np.arange(2.4, 14.5, delta_v.val[0])
bins3 = 60
# print(bins1)
# print(bins2)

# figtest, axtest = plt.subplots()
# num_bins = len(bins1) -1
# count_uncertainties = np.zeros(num_bins)
# print(bins1)

# for i in range(len(vx.val)):
#     val = vx.val[i]
#     err = vx.err[i]
    
#     # Find which bin index this data point belongs to (0-indexed)
#     bin_idx = np.digitize(val, bins1) - 1
#     #manual = (val - bins1[0]) // (bins1[1] - bins1[0])
#     #print(i, bin_idx, manual)
    
#     # Ignore points that fall exactly on the absolute outer limits
#     if bin_idx < 0 or bin_idx >= num_bins:
#         continue
        
#     # Check LEFT boundary: Does value - uncertainty cross into the previous bin?
#     if val - err < bins1[bin_idx]:
#         count_uncertainties[bin_idx] += 1        # Current bin is uncertain
#         if bin_idx - 1 >= 0:
#             count_uncertainties[bin_idx - 1] += 1 # Left neighbor is uncertain
            
#     # Check RIGHT boundary: Does value + uncertainty cross into the next bin?
#     if val + err > bins1[bin_idx + 1]:
#         count_uncertainties[bin_idx] += 1        # Current bin is uncertain
#         if bin_idx + 1 < num_bins:
#             count_uncertainties[bin_idx + 1] += 1 # Right neighbor is uncertain
# counts, bins = np.histogram(vx.val, bins=bins1)
# bin_centers = (bins[:-1] + bins[1:]) / 2
# bin_width = bins1[1] - bins1[0]
# print(bin_centers, counts, count_uncertainties)
# axtest.bar(bin_centers, counts, width=bin_width, color='skyblue', edgecolor='black', 
#        yerr=count_uncertainties, capsize=5, alpha=0.8, ecolor='red')
# # plt.close(figtest)
# plt.show()

ax.hist(vx.val, bins=bins1, edgecolor='black', label='naměřená data', density=True)
ax.set_xlabel('$v_x$ (mm/s)')
ax.set_ylabel('normovaná četnost')

mean, std = norm.fit(vx.val)
print(mean, std)
def gaussian(x, mean, stddev):
    return 1 / (stddev * np.sqrt(2 * np.pi)) * np.exp(-((x - mean)**2) / (2 * stddev**2))
counts, bins = np.histogram(vx.val, bins=bins1, density=True)
#counts = counts / np.sum(counts) #/ (bins[1] - bins[0])  # Normalize to get density
print(np.sum(counts))
bin_centers = (bins[:-1] + bins[1:]) / 2
# 4. Perform the fit
# p0 is the initial guess for [amplitude, mean, stddev]
initial_guess = [np.mean(vx.val), np.std(vx.val)]
popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
# 5. Extract Uncertainties
# The diagonal of pcov contains the variance of each parameter
perr = np.sqrt(np.diag(pcov))
mu, sigma = popt
mu_err, sigma_err = perr
#amp_var = pr.Var(amp, amp_err, 'A')
mu_var = pr.Var(mu, mu_err, '\\mu')
sigma_var = pr.Var(sigma, sigma_err, '\\sigma')
#pr.best_print(amp_var)
pr.best_print(mu_var)
pr.best_print(sigma_var)

x = np.linspace(bins[0], bins[-1], 100)
y = gaussian(x, *popt)
x_sigma = x[(x >= mu - std) & (x <= mu + std)]
y_sigma = y[(x >= mu - std) & (x <= mu + std)]
peak_y = gaussian(mu, *popt)

ax.plot(x, y, 'r-', label=f'fit normálního rozdělení')
ax.vlines(mu, 0, peak_y, linestyle='--', color='red', label=f'střední hodnota: ${mu_var}$ mm/s')
ax.fill_between(x_sigma, 0, y_sigma, alpha=0.25, color='red', 
    label=f'$\pm 1 \,\sigma$ interval: $\sigma = {sigma_var.val[0]:.1f}({sigma_var.err[0]*10:.0f})$ mm/s')
ax.margins(y=0.42)
ax.legend(loc='upper left')
fig.savefig('uloha18/histogram.png', dpi=300)
# plt.close(fig)



# now the same but with different bins
ax2.hist(vx.val, bins=bins2, edgecolor='black', label='naměřená data', density=True)
counts2, binsff = np.histogram(vx.val, bins=bins2, density=True)
#counts2 = counts2 / np.sum(counts2) / (binsff[1] - binsff[0])  # Normalize to get density
bin_centers2 = (binsff[:-1] + binsff[1:]) / 2
# 4. Perform the fit
# p0 is the initial guess for [amplitude, mean, stddev]
initial_guess2 = [np.mean(vx.val), np.std(vx.val)]
popt2, pcov2 = curve_fit(gaussian, bin_centers2, counts2, p0=initial_guess2)
# 5. Extract Uncertainties
# The diagonal of pcov contains the variance of each parameter
perr2 = np.sqrt(np.diag(pcov2))
mu2, sigma2 = popt2
mu_err2, sigma_err2 = perr2
#amp2, mu2, sigma2 = popt2
#amp_err2, mu_err2, sigma_err2 = perr2
#amp_var2 = pr.Var(amp2, amp_err2, 'A')
mu_var2 = pr.Var(mu2, mu_err2, '\\mu')
sigma_var2 = pr.Var(sigma2, sigma_err2, '\\sigma')
#pr.best_print(amp_var2)
pr.best_print(mu_var2)
pr.best_print(sigma_var2)
x2 = np.linspace(binsff[0], binsff[-1], 100)
y2 = gaussian(x2, *popt2)
x_sigma2 = x2[(x2 >= mu2 - sigma2) & (x2 <= mu2 + sigma2)]
y_sigma2 = y2[(x2 >= mu2 - sigma2) & (x2 <= mu2 + sigma2)]
peak_y2 = gaussian(mu2, *popt2)

ax2.plot(x2, y2, 'r-', label=f'fit normálního rozdělení')
ax2.vlines(mu2, 0, peak_y2, linestyle='--', color='red', label=f'střední hodnota: ${mu_var2}$ mm/s')
ax2.fill_between(x_sigma2, 0, y_sigma2, alpha=0.25, color='red', label=f'$\pm 1 \,\sigma$ interval: ${sigma_var2}$ mm/s')
ax2.margins(y=0.42)
ax2.legend(loc='upper left')
ax2.set_xlabel('$v_x$ (mm/s)')
ax2.set_ylabel('normovaná četnost')
fig2.savefig('uloha18/histogram2.png', dpi=300)
# plt.close(fig2)

plt.show()


print('All done!')