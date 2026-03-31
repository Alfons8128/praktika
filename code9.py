import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
from matplotlib.lines import Line2D

import praktika as pr


########################## glukoza #################################3
dfglukloza = pr.read_excel('uloha9/uloha9.ods', cells='A2:I20')
columns_old = dfglukloza.columns.to_list()
sigma_n = 0.0005
sigma_c = 0.5
sigma_z = 0.5
sigma_t = 0.4
nd_old = pr.Var(dfglukloza['nD'].to_numpy(), sigma_n, 'N_D')
c_old = pr.Var(dfglukloza[columns_old[0]].to_numpy(), sigma_c, 'c', '\\%')
z_old = pr.Var(dfglukloza[columns_old[2]].to_numpy(), sigma_z, 'Z')
t_old = pr.Var(dfglukloza[columns_old[8]].to_numpy(), sigma_t, 't', '\\degree C')

#print('Old values:')
#print(pr.to_table_2(c_old, nd_old, t_old))

# print(np.mean(nd_old.val[:3]), np.std(nd_old.val[:3], ddof=1), np.mean(nd_old.unc[:3]), pr.mean(nd_old.val[:3]))
# mean = (nd_old.val[0] + nd_old.val[1] + nd_old.val[2]) / 3
# mean_unc = (nd_old.unc[0] + nd_old.unc[1] + nd_old.unc[2]) / 3
# std_dev = np.sqrt(1/2 * ( (nd_old.val[0]-mean)**2 + (nd_old.val[1]-mean)**2 + (nd_old.val[2]-mean)**2) )
# print(mean, mean_unc, np.sqrt(3)/3*sigma_n, std_dev/np.sqrt(3), np.sqrt(std_dev**2 + sigma_n**2), pr.mean(nd_old.val[:3]).err)

nd_vals_all = dfglukloza['nD'].to_numpy()
nd_vals = np.array([np.mean(np.array(nd_vals_all[j:j+3])) for j in range(0, len(nd_vals_all), 3)])
t_vals = np.array([np.mean(np.array(t_old.val[j:j+3])) for j in range(0, len(t_old.val), 3)])
nd_errs_A = np.array([np.std(np.array(nd_vals_all[j:j+3])/np.sqrt(3), ddof=1) for j in range(0, len(nd_vals_all), 3)])
t_errs_A = np.array([np.std(np.array(t_old.val[j:j+3]), ddof=1)/np.sqrt(3) for j in range(0, len(t_old.val), 3)])
nd_errs = np.array([np.sqrt(err_A**2 + sigma_n**2) for err_A in nd_errs_A])
t_errs = np.array([np.sqrt(err_A**2 + sigma_t**2) for err_A in t_errs_A])

#print(nd_vals)
c = pr.NonErrorVar(c_old.val[::3], c_old.short_name, c_old.unit, fmt='.0f')
nd = pr.Var(nd_vals, nd_errs, 'N_D')
t = pr.Var(t_vals, t_errs, 't', '\\degree C')

dfglukoza_new = pr.read_excel('uloha9/uloha9.ods', cells='L2:M8')
a = pr.NonErrorVar(dfglukoza_new['A.1'].to_numpy(), 'A', fmt='.5f')
b = pr.NonErrorVar(dfglukoza_new['B.1'].to_numpy(), 'B', fmt='.5f')
sigma = pr.Var(-0.588, 0.020, '\\sigma')
delta = a + sigma * b
delta.set_lname('\\Delta')

#pr.best_print(delta)
#pr.best_print(c)
#print(c)
#pr.best_print(nd)
#pr.best_print(t)

#print(pr.to_table_2(c, nd, a, b, delta, t))
c = c / 100
c.set_lname(c.short_name, None)

index = pr.Rel(c, nd, pr.F.linear)
index.fit()#use_curve_fit=True)
#print(index.cov)
# print('index')
# for coeff in index.coeffs:
#     pr.best_print(coeff)

disperze = pr.Rel(c, delta, pr.F.linear)
disperze.fit()
# print('disperze')
# for coeff in disperze.coeffs:
#     pr.best_print(coeff)

fign, axn = plt.subplots(layout='constrained')
figd, axd = plt.subplots(layout='constrained')
index.plot_data(axn, err=(0,0))
index.plot_fit(axn, label='fitovaná lineární závislost',)
index.show_equation(axn, format='.7f')
pr.make_legend(axn, index)
disperze.plot_data(axd, label='naměřené hodnoty disperze',err=(0,1))
#disperze.plot_fit(axd, label='fitovaná lineární závislost')
#disperze.show_equation(axd, format='.7f')
pr.make_legend(axd, disperze)

fign.savefig('uloha9/glukoza_n.png', dpi=300)
figd.savefig('uloha9/glukoza_delta.png', dpi=300)
plt.close(fign)
plt.close(figd)

plt.show()

########################### polokoule a skla #########################

dfkalibrace = pr.read_excel('uloha9/uloha9.ods', cells='A23:B28')
cols = dfkalibrace.columns.to_list()
phi = pr.Var(dfkalibrace[cols[0]].to_numpy(), 2, '\\varphi', '\\degree')
alpha_val = dfkalibrace[cols[1]].to_numpy()
sigma_alpha_b = 1/60
bar_alpha = np.mean(alpha_val)
sigma_alpha_a = np.std(alpha_val)
sigma_alpha = np.sqrt(sigma_alpha_a**2 + sigma_alpha_b**2)
alpha_old = pr.NonErrorVar(alpha_val, '\\alpha_{m0}', '\\degree')
#print('prumer: ', bar_alpha)
#print('nejistota mezniho uhlu:', sigma_alpha)
#print(pr.to_table_2(phi, alpha_old))


dfpolokoule = pr.read_excel('uloha9/uloha9.ods', cells='K14:N26')
cols = dfpolokoule.columns.to_list()
phi = pr.Var(dfpolokoule[cols[0]].to_numpy(), 2, '\\varphi', '\\degree')
polokoule = pr.Var(dfpolokoule[cols[1]].to_numpy(), sigma_alpha, '\\alpha_{m0}', '\\degree')

mean_alpha = pr.mean(polokoule.val).add_sigma_b(sigma_alpha)
# print('prumer alpha: ', end='')
# pr.best_print(mean_alpha)
correction = mean_alpha - polokoule
mean_sin = pr.sin(mean_alpha.ensure_radians())
mean_n = 1 / mean_sin
mean_n.set_lname('N_1')

# print('real N_1')
# pr.best_print(mean_n)

sin_polokoule = pr.sin(polokoule.ensure_radians())
sin_polokoule.set_unit(None)
n_polokoule = 1 / sin_polokoule
n_polokoule.set_lname('N_1')

rel_polokoule = pr.Rel(phi, polokoule, pr.F.biased_sin)
rel_polokoule.fit([0.2, 34.9, -20], get_unit=False)
# print('alfa mezni polokoule')
# for coeff in rel_polokoule.coeffs:
#     pr.best_print(coeff)
fig_pol, ax_pol = plt.subplots()
rel_polokoule.plot_data(ax_pol, err=(0,1))
rel_polokoule.plot_fit(ax_pol, label='fitovaná sinová závislost')
#smooth_line = Line2D([], [], color='black', linestyle='--', linewidth=1, label='vodítko pro oko')
#rel_polokoule.handles.append(smooth_line)
pr.make_legend(ax_pol, rel_polokoule)
fig_pol.savefig('uloha9/polokoule.png', dpi=300)
plt.close(fig_pol)
#plt.show()

real_mean_alpha = rel_polokoule.coeffs[1].add_sigma_b(sigma_alpha)
#pr.best_print(real_mean_alpha)
real_mean_n = 1 / pr.sin(real_mean_alpha.ensure_radians())
#print('opravdu index polokoule')
real_mean_n.set_lname('N_1', None)
#pr.best_print(real_mean_n)

n_mean = np.mean(n_polokoule.val)
n_mean_sigma_a = np.std(n_polokoule.val, ddof=1)/np.sqrt(len(n_polokoule.val))  # nice, this works fine
#print('srovnani: ', n_mean_sigma_a, pr.mean(n_polokoule.val), pr.mean(n_polokoule), pr.mean(n_polokoule.val).err)
n_mean_sigma_b = 0.001
n_mean_sigma = np.sqrt(n_mean_sigma_a**2 + n_mean_sigma_b**2)
mean_polokoule = pr.Var(n_mean, n_mean_sigma, '\\bar{N}_1')
#mean_polokoule2 = mean_polokoule.add_sigma_b(0.001)
#pr.best_print(mean_polokoule)
#pr.best_print(pr.mean(n_polokoule.val).add_sigma_b(0.001))
#pr.best_print(mean_polokoule2)   # proof of no need to add sigma_b = 0.001

#pr.best_print(n_polokoule)
#pr.best_print(sin_polokoule)

#print(pr.to_table_2(phi, polokoule, sin_polokoule, n_polokoule))

######################## skla
dfskla = pr.read_excel(file_path='uloha9/uloha9.ods', cells='F49:G52')
alfa1 = pr.Var(dfskla['a1'].to_numpy(), sigma_alpha, '\\alpha_1', '\\degree')
alfa2 = pr.Var(dfskla['a2'].to_numpy(), sigma_alpha, '\\alpha_2', '\\degree')
alfa = (alfa1 + alfa2) /2

# pr.best_print(alfa1)
# pr.best_print(alfa2)
# pr.best_print(alfa)
n2 = real_mean_n * pr.sin(alfa.ensure_radians())
# pr.best_print(n2)


############################# kremen ###################################
dfkremen = pr.read_excel('uloha9/uloha9.ods', cells='M14:N26')
cols = dfkremen.columns.to_list()
radny_raw = pr.Var(dfkremen[cols[0]].to_numpy(), sigma_alpha, '\\alpha_o', '\\degree')
radny_mean = pr.mean(radny_raw.val).add_sigma_b(sigma_alpha)
print('radny mean')
pr.best_print(radny_mean)
correction2 = radny_mean - radny_raw
#correction3 = (correction2 + correction)/2
mimoradny_raw = pr.Var(dfkremen[cols[1]].to_numpy(), sigma_alpha, '\\alpha_e', '\\degree')


radny = radny_raw
mimoradny = mimoradny_raw

figraw, axraw = plt.subplots()
radny_rel = pr.Rel(phi, radny, pr.F.biased_sin)
radny_rel.fit([1.4, 61.9, -60], get_unit=False)
radny_rel.plot_data(axraw, label='naměřené hodnoty', err=(0,1))
radny_rel.plot_fit(axraw, label='fitovaná sinová závislost')
pr.make_legend(axraw, radny_rel)
figraw.savefig('uloha9/kremen_raw_angles.png', dpi=300)
plt.close(figraw)

print('křemen - radný paprsek - alfa mezni bar')
for coeff in radny_rel.coeffs:
    pr.best_print(coeff)
alfa_radny = radny_rel.coeffs[1]
alfa_radny = alfa_radny.add_sigma_b(sigma_alpha)
print('sigma alpha:', sigma_alpha)
pr.best_print(alfa_radny)
sin_radny = pr.sin(alfa_radny.ensure_radians())
n_radny = sin_radny * real_mean_n
n_radny.set_lname('N_o', None)
pr.best_print(n_radny)

correction3 = alfa_radny - radny
alfa_mimoradny = mimoradny + correction3


sin_radny = pr.sin(radny.ensure_radians())
sin_mimoradny = pr.sin(mimoradny.ensure_radians())
n_radny = sin_radny * real_mean_n
n_radny.set_lname('N_o', None)
n_mimoradny = sin_mimoradny * real_mean_n
n_mimoradny.set_lname('N_e', None)


radny2 = radny_raw + correction2
mimoradny2 = mimoradny_raw + correction2
sin_radny2 = pr.sin(radny2.ensure_radians())
sin_mimoradny2 = pr.sin(mimoradny2.ensure_radians())
n_radny2 = sin_radny2 * real_mean_n
n_radny2.set_lname('N_o', None)
n_mimoradny2 = sin_mimoradny2 * real_mean_n
n_mimoradny3 = pr.sin(alfa_mimoradny.ensure_radians()) * real_mean_n
n_mimoradny2.set_lname('N_e', None)
n_mimoradny3.set_lname('N_e', None)

print(pr.to_table_2(phi, alfa_mimoradny, n_mimoradny3))
# fig, ax = plt.subplots()
# rel = pr.Rel(phi, correction)
# rel2 = pr.Rel(phi, correction2, color='orange', shape='o')
# rel.plot_data(ax, label='polokoule')
# rel2.plot_data(ax, label='křemen')
# pr.make_legend(ax, rel, rel2)
#plt.close(fig)

#print(pr.to_table_2(phi, radny, n_radny, mimoradny, n_mimoradny))

figkremen, axkremen = plt.subplots()
radny_rel = pr.Rel(phi, n_radny)
mimoradny_rel = pr.Rel(phi, n_mimoradny, color='orange', shape='o')
radny_rel.plot_data(axkremen, label='řádný paprsek', err=(0,1))
mimoradny_rel.plot_data(axkremen, label='mimořádný paprsek', err=(0,1))
pr.make_legend(axkremen, radny_rel, mimoradny_rel)
axkremen.set_ylabel('N')
figkremen.savefig('uloha9/kremen_raw.png', dpi=300)
plt.close(figkremen)

figkremen2, axkremen2 = plt.subplots()
radny_rel2 = pr.Rel(phi, n_radny2)
mimoradny_rel2 = pr.Rel(phi, n_mimoradny3, pr.F.elliptic_degrees)
mimoradny_rel2.fit([1.543, 1.552, 20],get_unit=False)
#radny_rel2.plot_data(axkremen2, label='řádný paprsek', err=(0,1))
mimoradny_rel2.plot_data(axkremen2, label='index lomu po korekci', err=(0,1))

#print(mimoradny_rel2.handles)
mimoradny_rel2.plot_fit(axkremen2, label='fitovaná eliptická závislost')
#print(mimoradny_rel2.handles)
#mimoradny_rel2.show_equation(axkremen2)
#print(mimoradny_rel2.handles)
for coeff in mimoradny_rel2.coeffs:
    pr.best_print(coeff)

pr.make_legend(axkremen2, radny_rel2, mimoradny_rel2, loc='upper left')
#axkremen2.legend(loc='upper left')
axkremen2.set_ylabel('N')
figkremen2.savefig('uloha9/kremen_corrected.png', dpi=300)
plt.close(figkremen2)

plt.show()

print('All done.')