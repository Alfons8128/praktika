import numpy as np
import praktika as pr
import matplotlib.pyplot as plt

dfglukoza = pr.read_excel('uloha11/uloha11.ods', cells='B2:C8')
gluk_cols = dfglukoza.columns.to_list()
dfgluk_unc = pr.read_excel('uloha11/uloha11.ods', cells='A11:F26')
gluk_unc_cols = dfgluk_unc.columns.to_list()
dfbenzen = pr.read_excel('uloha11/uloha11.ods', cells='A35:D48')
dfbenzen_unc = pr.read_excel('uloha11/uloha11.ods', cells='E35:H50')

koncentrace = pr.NonErrorVar(dfgluk_unc[gluk_unc_cols[0]], short_name='c', unit='g/l', fmt='.0f')
cislo_mereni = pr.NonErrorVar(dfgluk_unc['cislo'], short_name='číslo měření', fmt='.0f')
alfa_kalib = pr.NonErrorVar(dfgluk_unc[gluk_unc_cols[2]], short_name='\\alpha', unit='\\degree', fmt='.2f')
alfa_kalib_mean = pr.mean(alfa_kalib.val)
alfa_kalib_unc = pr.std(alfa_kalib)

#pr.best_print(alfa_kalib_mean)
#print(alfa_kalib_unc)
cislo1 = pr.NonErrorVar(cislo_mereni.val[:5], short_name='číslo měření', fmt='.0f')
cislo2 = pr.NonErrorVar(cislo_mereni.val[5:10], short_name='číslo měření', fmt='.0f')
cislo3 = pr.NonErrorVar(cislo_mereni.val[10:15], short_name='číslo měření', fmt='.0f')
alfa1 = pr.NonErrorVar(alfa_kalib.val[:5], short_name='\\alpha', unit='\\degree', fmt='.2f')
alfa2 = pr.NonErrorVar(alfa_kalib.val[5:10], short_name='\\alpha', unit='\\degree', fmt='.2f')
alfa3 = pr.NonErrorVar(alfa_kalib.val[10:15], short_name='\\alpha', unit='\\degree', fmt='.2f')

#print(pr.to_table_2(cislo1,alfa1, alfa2, alfa3))

sigma_beta = alfa_kalib_unc
sigma_v = 0.016
sigma_c_num = np.sqrt(2) * sigma_v
sigma_c = f'{sigma_c_num*100}%'
#print(sigma_c)
concentration = pr.Var(dfglukoza[gluk_cols[0]], sigma_c, short_name='c', unit='g/l')
beta = pr.Var(dfglukoza[gluk_cols[1]], sigma_beta, short_name='\\beta', unit='\\degree')
beta0 = pr.Var(dfglukoza[gluk_cols[1]][0], sigma_beta, short_name='\\beta_0', unit='\\degree')

alfa = beta - beta0
alfa.set_lname('\\alpha', '\\degree')
#pr.best_print(alfa)

alfa2 = pr.Var(alfa.val[1:], alfa.err[1:], short_name='\\alpha', unit='\\degree')
concentration2 = pr.Var(concentration.val[1:], concentration.err[1:], short_name='c', unit='g/l')
rho = alfa2 / concentration2
rho.set_lname('\\rho', '\\degree dm^2 g^{-1}')
rho2 = pr.Var(list([0,*rho.val]), list([0.01,*rho.err]), short_name='\\rho', unit='\\degree dm^2 g^{-1}')
#pr.best_print(rho2)
mean_rho = pr.mean(rho.val)
mean_std_rho = np.mean(rho.err)
mean_rho = mean_rho.add_sigma_b(mean_std_rho)
#pr.best_print(mean_rho)


#print(pr.to_table_2(concentration, beta, alfa, rho2))

rel_glukoza = pr.Rel(concentration, beta, pr.F.linear)
figgluk, axgluk = plt.subplots()
rel_glukoza.fit(absolute_sigma=True)
# print('Fit glukoza:')
# for coeff in rel_glukoza.coeffs:
#     pr.best_print(coeff)
rel_glukoza.plot_data(axgluk, err=(1,0))
rel_glukoza.plot_fit(axgluk)
rel_glukoza.show_equation(axgluk, ['.2f','.4f'])
rel_glukoza.make_legend(axgluk)


figgluk.savefig('uloha11/glukoza.png', dpi=300)
plt.close(figgluk)

######################### benzen #############################

bunc_cols = dfbenzen_unc.columns.to_list()
cislo = pr.NonErrorVar(dfbenzen_unc[bunc_cols[1]][:5], 'cislo mereni', fmt='.0f')
bunc1 = pr.NonErrorVar(dfbenzen_unc[bunc_cols[2]][:5], '\\beta', '\\degree', fmt='.2f')
bunc2 = pr.NonErrorVar(dfbenzen_unc[bunc_cols[2]][5:10], '\\beta', '\\degree', fmt='.2f')
bunc3 = pr.NonErrorVar(dfbenzen_unc[bunc_cols[2]][10:], '\\beta', '\\degree', fmt='.2f')

#print(pr.to_table_2(cislo, bunc1, bunc2, bunc3))
mean1 = pr.mean(bunc1.val)
std1 = pr.std(bunc1.val)
mean2 = pr.mean(bunc2.val)
std2 = pr.std(bunc2.val)
mean3 = pr.mean(bunc3.val)
std3 = pr.std(bunc3.val)

#print(mean1, std1, mean2, std2, mean3, std3)
sigma_beta_benzen = (std1 + std2 + std3)/3

bcols = dfbenzen.columns.to_list()
beta_benzen = pr.Var(dfbenzen[bcols[1]].to_numpy(), sigma_beta_benzen, '\\beta', '\\degree')
i_values = dfbenzen[bcols[0]].to_numpy()
i_errors = [0.002 * abs(I) + 0.001 for I in i_values]

i_benzen = pr.Var([i_values[i] for i in range(len(i_values)-1, -1, -1)], i_errors, 'I', 'A')
pr.best_print(i_benzen)

dffluke = pr.read_excel('uloha11/uloha11.ods', cells='J44:M46')
fluke = pr.MeasureUnc('range', 'A', dffluke)

#i_benzen = fluke.set_uncertainty(i_benzen)
#pr.best_print(i_benzen)
beta0_benzen = pr.Var(beta_benzen.val[6], sigma_beta_benzen, '\\beta', '\\degree')
alfa_benzen = beta_benzen - beta0_benzen
b_benzen = 0.0138 * i_benzen
b_benzen.set_lname('B', 'T')
#pr.best_print(b_benzen)
d2 = 2

b2 = pr.Var(list(b_benzen.val[:6]) + list(b_benzen.val[7:]), list(b_benzen.err[:6]) + list(b_benzen.err[7:]), 'B', 'T')
a2 = pr.Var(list(alfa_benzen.val[:6]) + list(alfa_benzen.val[7:]), list(alfa_benzen.err[:6]) + list(alfa_benzen.err[7:]), '\\alpha', '\\degree')
#pr.best_print(b2)
v2 = a2 / b2 / d2
v2.set_lname('V', '\\degree T^{-1} dm^{-1}')

verdet = pr.Var(list(v2.val[:6]) + [0] + list(v2.val[6:]), list(v2.err[:6]) + [0.001] + list(v2.err[6:]), 'V', v2.unit)
verdet_mean = pr.mean(v2.val)
verdet_weighted_mean = pr.mean(v2)
print('mean verdet: ', end='')
pr.best_print(verdet_mean)
pr.best_print(verdet_weighted_mean)
print(verdet_weighted_mean.err)

####################
figv,axv = plt.subplots()
rel_verdet = pr.Rel(b2, v2, pr.F.const)
rel_verdet.fit(absolute_sigma=False)
#print('constant fit verdet: ', end='')
#print(rel_verdet.chi)
#print(rel_verdet.coeffs[0].err)
#print(rel_verdet.coeffs[0].err/np.sqrt(rel_verdet.chi))
pr.best_print(rel_verdet.coeffs[0])

rel_verdet.plot_data(axv)
rel_verdet.plot_fit(axv)
rel_verdet.show_equation(axv)
rel_verdet.make_legend(axv)
plt.close(figv)
#####################

#print(pr.to_table_2(i_benzen, b_benzen, beta_benzen, alfa_benzen, verdet))

figbenz, axbenz = plt.subplots()

rel_benzen = pr.Rel(b_benzen, beta_benzen, pr.F.linear)
rel_benzen.fit(absolute_sigma=False)
print('fit benzen:')
print(rel_benzen.chi)
for coeff in rel_benzen.coeffs:
    pr.best_print(coeff)
verdet_fit = rel_benzen.coeffs[1] / 2
verdet_fit.set_lname('V', v2.unit)
print('fitted verdet: ', end='')
pr.best_print(verdet_fit)

rel_benzen.plot_data(axbenz, err=(0,0))
rel_benzen.plot_fit(axbenz)
rel_benzen.show_equation(axbenz, format=['.3f', '.1f'])
rel_benzen.make_legend(axbenz)

figbenz.savefig('uloha11/benzen.png', dpi=300)
plt.close(figbenz)
plt.show()

print('All done.')