import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import praktika as pr

filename = 'uloha2/uloha2.ods'
sigmax = 0.1
sigmay = 0.01
sigmay = sigmay/np.sqrt(5)

dfempty = pr.read_excel(filename, cells='O1:O25')
ecols = dfempty.columns.tolist()
placeholder = pr.NonErrorVar(dfempty[ecols[0]], fmt='.0f')

dff1 = pr.read_excel(filename, cells='A5:F11')
dff2 = pr.read_excel(filename, cells='A16:F22')
dff3 = pr.read_excel(filename, cells='A27:F33')
dff4 = pr.read_excel(filename, cells='A38:F44')
cols1 = dff1.columns.tolist()
cols2 = dff2.columns.tolist()
cols3 = dff3.columns.tolist()
cols4 = dff4.columns.tolist()

d1 = pr.Var(dff1[cols1[0]].to_list()+dff2[cols2[0]].to_list()+dff3[cols3[0]].to_list()+dff4[cols4[0]].to_list(), sigmax,
            short_name='d_1', unit='cm')
d2 = pr.Var(dff1[cols1[1]].to_list()+dff2[cols2[1]].to_list()+dff3[cols3[1]].to_list()+dff4[cols4[1]].to_list(), sigmax,
            short_name='d_2', unit='cm')
yl1 = pr.Var(dff1[cols1[2]].to_list()+dff2[cols2[2]].to_list()+dff3[cols3[2]].to_list()+dff4[cols4[2]].to_list(), sigmax,
            short_name='y_{l1}', unit='cm')
yr1 = pr.Var(dff1[cols1[3]].to_list()+dff2[cols2[3]].to_list()+dff3[cols3[3]].to_list()+dff4[cols4[3]].to_list(), sigmax,
            short_name='y_{r1}', unit='cm')
yl2 = pr.Var(dff1[cols1[4]].to_list()+dff2[cols2[4]].to_list()+dff3[cols3[4]].to_list()+dff4[cols4[4]].to_list(), sigmax,
            short_name='y_{l2}', unit='cm')
yr2 = pr.Var(dff1[cols1[5]].to_list()+dff2[cols2[5]].to_list()+dff3[cols3[5]].to_list()+dff4[cols4[5]].to_list(), sigmax,
            short_name='y_{r2}', unit='cm')


#print(pr.to_table_2(placeholder, d1, yl1, yr1, d2, yl2, yr2))
x1 = pr.Var(np.array([pr.mean(d1.val[:5]).add_sigma_b(sigmax).val[0], pr.mean(d1.val[6:11]).add_sigma_b(sigmax).val[0], 
             pr.mean(d1.val[12:17]).add_sigma_b(sigmax).val[0], pr.mean(d1.val[18:23]).add_sigma_b(sigmax).val[0]]),
             np.array([pr.mean(d1.val[:5]).add_sigma_b(sigmax).err[0], pr.mean(d1.val[6:11]).add_sigma_b(sigmax).err[0], 
              pr.mean(d1.val[12:17]).add_sigma_b(sigmax).err[0], pr.mean(d1.val[18:23]).add_sigma_b(sigmax).err[0]]), short_name='x', unit='cm')
yl2 = pr.Var(np.array([pr.mean(yl2.val[:5]).add_sigma_b(sigmay).val[0], pr.mean(yl2.val[6:11]).add_sigma_b(sigmay).val[0], 
              pr.mean(yl2.val[12:17]).add_sigma_b(sigmay).val[0], pr.mean(yl2.val[18:23]).add_sigma_b(sigmay).val[0]]),
             np.array([pr.mean(yl2.val[:5]).add_sigma_b(sigmay).err[0], pr.mean(yl2.val[6:11]).add_sigma_b(sigmay).err[0], 
              pr.mean(yl2.val[12:17]).add_sigma_b(sigmay).err[0], pr.mean(yl2.val[18:23]).add_sigma_b(sigmay).err[0]]), short_name='y_{l}', unit='cm')
yl2 = yl2/10
yr2 = pr.Var(np.array([pr.mean(yr2.val[:5]).add_sigma_b(sigmay).val[0], pr.mean(yr2.val[6:11]).add_sigma_b(sigmay).val[0], 
              pr.mean(yr2.val[12:17]).add_sigma_b(sigmay).val[0], pr.mean(yr2.val[18:23]).add_sigma_b(sigmay).val[0]]),
             np.array([pr.mean(yr2.val[:5]).add_sigma_b(sigmay).err[0], pr.mean(yr2.val[6:11]).add_sigma_b(sigmay).err[0], 
              pr.mean(yr2.val[12:17]).add_sigma_b(sigmay).err[0], pr.mean(yr2.val[18:23]).add_sigma_b(sigmay).err[0]]), short_name='y_{r2}', unit='cm')
yr2 =yr2/10
x2 = pr.Var(np.array([pr.mean(d2.val[:5]).add_sigma_b(sigmax).val[0], pr.mean(d2.val[6:11]).add_sigma_b(sigmax).val[0], 
             pr.mean(d2.val[12:17]).add_sigma_b(sigmax).val[0], pr.mean(d2.val[18:23]).add_sigma_b(sigmax).val[0]]),
             np.array([pr.mean(d2.val[:5]).add_sigma_b(sigmax).err[0], pr.mean(d2.val[6:11]).add_sigma_b(sigmax).err[0], 
              pr.mean(d2.val[12:17]).add_sigma_b(sigmax).err[0], pr.mean(d2.val[18:23]).add_sigma_b(sigmax).err[0]]), short_name='x', unit='cm')
yl1 = pr.Var(np.array([pr.mean(yl1.val[:5]).add_sigma_b(sigmay).val[0], pr.mean(yl1.val[6:11]).add_sigma_b(sigmay).val[0], 
              pr.mean(yl1.val[12:17]).add_sigma_b(sigmay).val[0], pr.mean(yl1.val[18:23]).add_sigma_b(sigmay).val[0]]),
             np.array([pr.mean(yl1.val[:5]).add_sigma_b(sigmay).err[0], pr.mean(yl1.val[6:11]).add_sigma_b(sigmay).err[0], 
              pr.mean(yl1.val[12:17]).add_sigma_b(sigmay).err[0], pr.mean(yl1.val[18:23]).add_sigma_b(sigmay).err[0]]), short_name='y_{l}', unit='cm')
yl1 = yl1/10
yr1 = pr.Var(np.array([pr.mean(yr1.val[:5]).add_sigma_b(sigmay).val[0], pr.mean(yr1.val[6:11]).add_sigma_b(sigmay).val[0], 
              pr.mean(yr1.val[12:17]).add_sigma_b(sigmay).val[0], pr.mean(yr1.val[18:23]).add_sigma_b(sigmay).val[0]]),
             np.array([pr.mean(yr1.val[:5]).add_sigma_b(sigmay).err[0], pr.mean(yr1.val[6:11]).add_sigma_b(sigmay).err[0],
              pr.mean(yr1.val[12:17]).add_sigma_b(sigmay).err[0], pr.mean(yr1.val[18:23]).add_sigma_b(sigmay).err[0]]), short_name='y_{r}', unit='cm')
yr1 = yr1/10
#print(x1, yl1, yr1, x2, yl2, yr2, sep='\n')
y1 = yr1-yl1
y1.set_lname("Y'", 'cm')
y2 = yr2-yl2
y2.set_lname("Y'", 'cm')
d = pr.Var([84, 94, 104, 89], sigmax, short_name='D', unit='cm')
delta = x2-x1
delta.set_lname('\\Delta', 'cm')
vzory = pr.Var([1,1,1,1], 0.01, short_name='Y', unit='cm')

ykuy1 = y1 / vzory
ykuy2 = y2 / vzory
ykuy1.set_lname("\\frac{Y'}{{Y}}", None)
ykuy2.set_lname("\\frac{Y'}{{Y}}", None)
# pr.best_print(ykuy1)
# print(pr.to_table_2(d, x1, y1, ykuy1, x2, y2, ykuy2, delta))

fbessel = (d**2 - delta**2) / (4*d)
fbessel.set_lname('f_{Bessel}', 'cm')
fdvoji = delta / (ykuy1 - ykuy2)
fdvoji.set_lname('f_{dvoji}', 'cm')

#print(pr.to_table_2(d, fbessel, fdvoji))

'''
bessel_mean1 = pr.mean(fbessel.val).add_sigma_b(pr.mean(fbessel.err).val[0])
bessel_mean2 = pr.weighted_mean(fbessel)
bessel_mean3 = pr.weighted_mean(fbessel, absolute_sigma=False)
error4 = 1/4 * np.sqrt(np.sum(np.array([err**2 for err in fbessel.err]))) # nejistota jednotlivych mereni (bez rozptylu)
error5 = np.sqrt(error4**2 + pr.mean(fbessel.val).err[0]**2) 
bessel_mean6 = pr.mean(fbessel)
bessel_mean7 = pr.mean(fbessel.val).add_sigma_b(error4)
print(bessel_mean6.err[0], bessel_mean7.err[0])
#pr.best_print(bessel_mean1)
#pr.best_print(bessel_mean2)
#pr.best_print(bessel_mean3)
print(error4)
print(error5)

fig1, ax1 = plt.subplots()
rel1 = pr.Rel(d, fbessel, pr.F.const)
rel1.fit(absolute_sigma=False)
#pr.best_print(rel1.coeffs[0])
rel1.plot_data(ax1)
rel1.plot_fit(ax1)
rel1.show_equation(ax1)
rel1.make_legend(ax1)
plt.close(fig1)
plt.show()
#'''

bessel_mean = pr.mean(fbessel)
dvoji_mean = pr.mean(fdvoji)
#pr.best_print(bessel_mean)
#pr.best_print(dvoji_mean)

sigma_d = 0.01 # cm
dfclona = pr.read_excel(filename, cells='K18:N23')
clona_cols = dfclona.columns.tolist()
cislo_clony = pr.NonErrorVar(dfclona[clona_cols[0]], short_name='clona', fmt='.0f')
vnejsi = pr.Var(dfclona[clona_cols[1]], sigma_d, short_name='d_2', unit='cm')
vnitrni = pr.Var(dfclona[clona_cols[2]], sigma_d, short_name='d_1', unit='cm')
s = (vnejsi + vnitrni) / 4
s.set_lname('s', 'cm')
#pr.best_print(s)
#print(pr.to_table_2(cislo_clony, vnejsi, vnitrni, s))

dfplus30 = pr.read_excel(filename, cells='K3:N8')
dfminus30 = pr.read_excel(filename, cells='K11:N16')
dfplus60 = pr.read_excel(filename, cells='P3:S8')
dfminus60 = pr.read_excel(filename, cells='P11:S16')
plus30cols = dfplus30.columns.to_list()
minus30cols = dfminus30.columns.to_list()
plus60cols = dfplus60.columns.to_list()
minus60cols = dfminus60.columns.to_list()

clony_values = list(dfplus30[plus30cols[0]])+list(dfminus30[minus30cols[0]])+list(dfplus60[plus60cols[0]])+list(dfminus60[minus60cols[0]])
#print(clony_values)
clona = pr.NonErrorVar(clony_values, 'clona', fmt='.0f')
xprime_values = list(dfplus30[plus30cols[1]])+list(dfminus30[minus30cols[1]])+list(dfplus60[plus60cols[1]])+list(dfminus60[minus60cols[1]])
xprime = pr.Var(xprime_values, sigmax, "x'", 'cm')
aprime_values = list(dfplus30[plus30cols[2]])+list(dfminus30[minus30cols[2]])+list(dfplus60[plus60cols[2]])+list(dfminus60[minus60cols[2]])
aprime = pr.Var(aprime_values, sigmax, "a'", 'cm')
delta_a_values = list(dfplus30[plus30cols[3]])+list(dfminus30[minus30cols[3]])+list(dfplus60[plus60cols[3]])+list(dfminus60[minus60cols[3]])
delta_a = pr.Var(delta_a_values, np.sqrt(2)*sigmax, "\\Delta a'", 'cm')

clona = pr.NonErrorVar(dfplus30[plus30cols[0]].to_numpy(),'clona', fmt='.0f')
aprimeplus30 = pr.Var(dfplus30[plus30cols[2]], sigmax*2, "a'", 'cm')
delta_a_plus30 = pr.Var(dfplus30[plus30cols[3]], np.sqrt(2)*sigmax*2, "\\Delta a'", 'cm')
aprimeplus60 = pr.Var(dfplus60[plus60cols[2]], sigmax*2, "a'", 'cm')
delta_a_plus60 = pr.Var(dfplus60[plus60cols[3]], np.sqrt(2)*sigmax*2, "\\Delta a'", 'cm')
aprimeminus30 = pr.Var(dfminus30[minus30cols[2]], sigmax*2, "a'", 'cm')
delta_a_minus30 = pr.Var(dfminus30[minus30cols[3]], np.sqrt(2)*sigmax*2, "\\Delta a'", 'cm')
aprimeminus60 = pr.Var(dfminus60[minus60cols[2]], sigmax*2, "a'", 'cm')
delta_a_minus60 = pr.Var(dfminus60[minus60cols[3]], np.sqrt(2)*sigmax*2, "\\Delta a'", 'cm')

#print(pr.to_table_2(clona, aprimeplus30, delta_a_plus30, aprimeplus60, delta_a_plus60, aprimeminus30, delta_a_minus30, aprimeminus60, delta_a_minus60))
s2 = s**2
s2.set_lname('s^2', 'cm^2')

figkul, axkul = plt.subplots()
relplus30 = pr.Rel(s2, delta_a_plus30, pr.F.direct)
relplus60 = pr.Rel(s2, delta_a_plus60, pr.F.direct, color='blue', shape='o')
relminus30 = pr.Rel(s2, delta_a_minus30, pr.F.direct, color='red', shape='^')
relminus60 = pr.Rel(s2, delta_a_minus60, pr.F.direct, color='green', shape='v')
rels = [relplus30, relplus60, relminus30, relminus60]
for i,rel in enumerate(rels):
    if i != 6:
        rel.fit(absolute_sigma=True)
    else:
        rel.fit(absolute_sigma=False)
    for coeff in rel.coeffs:
        pass
        #pr.best_print(coeff)
    #print(rel.chi)
relplus30.plot_data(axkul, err=(0,1), label='vypuklá 30 cm')
relplus60.plot_data(axkul, err=(0,1), label='vypuklá 60 cm')
relminus30.plot_data(axkul, err=(0,1), label='plochá 30 cm')
relminus60.plot_data(axkul, err=(0,1), label='plochá 60 cm')

relplus30.plot_fit(axkul, 'vypuklá 30 lineární fit')
relplus60.plot_fit(axkul, 'vypuklá 60 lineární fit')
relminus30.plot_fit(axkul, 'plochá 30 lineární fit')
relminus60.plot_fit(axkul, 'plochá 60 lineární fit')

for rel in rels:
    rel.show_equation(axkul, format='.2f', newline=False, combined=True)

pr.make_legend(axkul, relplus60, relminus60, relminus30, relplus30)

figkul.savefig('uloha2/kulova_vada.png', dpi=300)
plt.close(figkul)
plt.show()

smernice_val = [rel.coeffs[0].val[0] for rel in rels]
smernice_err = [rel.coeffs[0].err[0] for rel in rels]
smernice = pr.Var(smernice_val, smernice_err, 'K', 'cm^{-1}')
place = pr.NonErrorVar(np.ones_like(smernice_val), 'place', fmt='.0f')
#pr.best_print(smernice)
#print(pr.to_table_2(place, smernice))

################ fokometr
phi = pr.Var(5.25, 0.25, '\\varphi', 'D')
#pr.best_print(phi)
f_fokometr = 1/phi * 100
f_fokometr.set_lname('f_{fokometr}', 'cm')
#pr.best_print(f_fokometr)


################# tlouska 
df_tlouska = pr.read_excel(filename, cells='C47:G52')
tlouska_cols = df_tlouska.columns.to_list()
cislo = pr.NonErrorVar(df_tlouska[tlouska_cols[0]], 'číslo měření', fmt='.0f')
sigma_t = 0.01
t6 = pr.Var(df_tlouska[tlouska_cols[1]], sigma_t, 't', 'mm')
t2 = pr.Var(df_tlouska[tlouska_cols[4]], sigma_t, 't', 'mm')
mean_t6 = pr.mean(t6)
mean_t2 = pr.mean(t2)

#print(pr.to_table_2(cislo, t2, t6))
#pr.best_print(mean_t6)
#pr.best_print(mean_t2)

df_roviny = pr.read_excel(filename, cells='T46:W48')
#print(pr.excel_to_latex_2(filename, cells='T46:W48', format='.1f'))
delta2 = pr.Var(2.2, 0.4, '\\delta', 'mm')
delta6 = pr.Var(10.8, 0.4, '\\delta', 'mm')

cocky = pr.NonErrorVar([2, 6], 'čočka', fmt='.0f')
n = mean_t6 / (mean_t6 - delta6)
#pr.best_print(n)
dprime = d - delta2/10
f_bessel_2 = (dprime**2 - delta**2) / (4*dprime)
f_bessel_2.set_lname('f_{Bessel}\'', 'cm')
f_bessel_2_mean = pr.mean(f_bessel_2)
pr.best_print(f_bessel_2_mean)
info = pr.NonErrorVar(0, 'metoda', fmt='.0f')
print(pr.to_table_2(info, dvoji_mean, bessel_mean, f_bessel_2_mean, f_fokometr))

print('All done.')