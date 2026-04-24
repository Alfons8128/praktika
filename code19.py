import numpy as np
import matplotlib.pyplot as plt
import praktika as pr
from uncertainties import umath as um
import pandas as pd

filename = 'uloha19/uloha19.ods'

df12 = pr.read_excel(filename, cells='A2:F26')
cols12 = df12.columns.tolist()
df3 = pr.read_excel(filename, cells='A29:C53')
cols3 = df3.columns.tolist()
df45 = pr.read_excel(filename, cells='H2:M26')
cols45 = df45.columns.tolist()

k = pr.NonErrorVar(df12[cols12[0]], 'k', fmt='.0f')
lambda_Na = pr.Var(589.3, 0.3, 'lambda_Na', 'nm')
#lambda_Na = pr.NonErrorVar(589.3, 'lambda_Na', fmt='.1f')
l = pr.Var(50, 0.05, 'l', 'cm')

n = 1 + k * lambda_Na/l * 1e-7
n.err[0] = 1e-8
n.set_lname('n', None)
# pr.best_print(k)
# pr.best_print(n)

sigma_p = 2
p1 = pr.Var(df12[cols12[1]], sigma_p, 'p', 'hPa')
p2 = pr.Var(df12[cols12[4]], sigma_p, 'p', 'hPa')
p3 = pr.Var(df3[cols3[1]], sigma_p, 'p', 'hPa')
p4 = pr.Var(df45[cols45[1]], sigma_p, 'p', 'hPa')
p5 = pr.Var(df45[cols45[4]], sigma_p, 'p', 'hPa')
# for p in [p1, p2, p3, p4, p5]:
#     pr.best_print(p)

#print(pr.to_table_2(k, n, p1, p2, p3, p4, p5))


############################## fitting #############################
def index(p, tau):
    return 1 + tau * p

n_minus1 = n - 1
n_minus1.err[0] = 1e-8
n_minus1.set_lname('n-1', None)
#pr.best_print(n_minus1)
# print(n_minus1.err[0])

# rel1 = pr.Rel(p1, n_minus1, pr.F.index2)
# rel2 = pr.Rel(p2, n_minus1, pr.F.index2, color='orange', shape='o')
# rel3 = pr.Rel(p3, n_minus1, pr.F.index2, color='green', shape='d')
# rel4 = pr.Rel(p4, n_minus1, pr.F.index2, color='red', shape='^')
# rel5 = pr.Rel(p5, n_minus1, pr.F.index2, color='blue', shape='v')

rel1 = pr.Rel(p1, n, pr.F.index)
rel2 = pr.Rel(p2, n, pr.F.index, color='orange', shape='o')
rel3 = pr.Rel(p3, n, pr.F.index, color='green', shape='d')
rel4 = pr.Rel(p4, n, pr.F.index, color='red', shape='^')
rel5 = pr.Rel(p5, n, pr.F.index, color='blue', shape='v')

rel6 = pr.Rel(n, p1, pr.F.index_inv)
rel7 = pr.Rel(n, p2, pr.F.index_inv, color='orange', shape='o')
rel8 = pr.Rel(n, p3, pr.F.index_inv, color='green', shape='d')
rel9 = pr.Rel(n, p4, pr.F.index_inv, color='red', shape='^')
rel10 = pr.Rel(n, p5, pr.F.index_inv, color='blue', shape='v')

rels = [rel1, rel2, rel3, rel4, rel5]
rels_inv = [rel6, rel7, rel8, rel9, rel10]
for i,rel in enumerate(rels):
    rel.fit(p0=[2.65e-7], absolute_sigma=False, get_unit=False)
    # print(f"Rel {i+1}: {rel.chi}")
    # for coef in rel.coeffs:
    #     pr.best_print(coef)

for i,rel in enumerate(rels_inv):
    rel.fit(p0=[2.65e-7], absolute_sigma=True, get_unit=False)
    # print(f"Rel {i+6}: {rel.chi}")
    # for coef in rel.coeffs:
    #     pr.best_print(coef)

# for i in range(len(rels)):
#     print(f'rel{i+1}: ',end='')
#     pr.best_print(rels[i].coeffs[0])
#     print(f'rel{i+6}: ',end='')
#     pr.best_print(rels_inv[i].coeffs[0])

taus = pr.Var([rel.coeffs[0].val[0] for rel in rels], [rel.coeffs[0].err[0] for rel in rels], 'tau', '1/hPa')
#pr.best_print(taus)
tau = pr.mean(taus)
count = pr.NonErrorVar([1,2,3,4,5], 'měření', fmt='.0f')
tau.set_lname('\\tau', '1/hPa')
#pr.best_print(tau)
#print(pr.to_table_2(count, taus))
############################## plotting #############################
figinv, axinv = plt.subplots(layout='constrained')
rel6.plot_data(axinv, label=f'měření 1', err=(1,1))
eq_string = f'$p = {rel6.coeffs[0].val[0]*1e7:.3f}\\cdot 10^{{-7}} (n - 1)$'
rel6.plot_fit(axinv, label=f'lineární fit měření 1')
rel6.show_equation(axinv, format='.2e', combined=True, eq_string=eq_string)
axinv.ticklabel_format(style='plain', axis='x', useOffset=False)
pr.make_legend(axinv, rel6)
plt.close(figinv)
plt.show()

figall, axall = plt.subplots(layout='constrained')
for i,rel in enumerate(rels):
    rel.plot_data(axall, label=f'měření {i+1}', err=(0,0))
    eq_string = f'${rel.y.short_name} = 1 + {rel.coeffs[0].val[0]*1e7:.3f}\\cdot 10^{{{-7}}} {rel.x.short_name}$'
    rel.plot_fit(axall, label=f'lineární fit měření {i+1}')
    rel.show_equation(axall, format='.2e', combined=True, eq_string=eq_string)
axall.ticklabel_format(style='plain', axis='y', useOffset=False)

pr.make_legend(axall, rel1, rel2, rel3, rel4, rel5)
figall.savefig('uloha19/index_na_tlaku_all.png', dpi=300)
plt.close(figall)

fig1, ax1 = plt.subplots(layout='constrained')
rel1.plot_data(ax1, err=(0,0))
eq_string = f'${rel1.y.short_name} = 1 + {rel1.coeffs[0].val[0]*1e7:.3f}\\cdot 10^{{{-7}}} {rel1.x.short_name}$'
rel1.plot_fit(ax1, label='lineární fit')
rel1.show_equation(ax1, combined=False, eq_string=eq_string, newline=False)
ax1.ticklabel_format(style='plain', axis='y', useOffset=False)
pr.make_legend(ax1, rel1)
fig1.savefig('uloha19/index_na_tlaku_1.png', dpi=300)

fig2, ax2 = plt.subplots(layout='constrained')
rel2.plot_data(ax2, err=(0,0))
eq_string = f'${rel2.y.short_name} = 1 + {rel2.coeffs[0].val[0]*1e7:.3f}\\cdot 10^{{{-7}}} {rel2.x.short_name}$'
rel2.plot_fit(ax2, label='lineární fit')
rel2.show_equation(ax2, combined=False, eq_string=eq_string, newline=False)
ax2.ticklabel_format(style='plain', axis='y', useOffset=False)
pr.make_legend(ax2, rel2)
fig2.savefig('uloha19/index_na_tlaku_2.png', dpi=300)

fig3, ax3 = plt.subplots(layout='constrained')
rel3.plot_data(ax3, err=(0,0))
eq_string = f'${rel3.y.short_name} = 1 + {rel3.coeffs[0].val[0]*1e7:.3f}\\cdot 10^{{{-7}}} {rel3.x.short_name}$'
rel3.plot_fit(ax3, label='lineární fit')
rel3.show_equation(ax3, combined=False, eq_string=eq_string, newline=False)
ax3.ticklabel_format(style='plain', axis='y', useOffset=False)
pr.make_legend(ax3, rel3)
fig3.savefig('uloha19/index_na_tlaku_3.png', dpi=300)

fig4, ax4 = plt.subplots(layout='constrained')
rel4.plot_data(ax4, err=(0,0))
eq_string = f'${rel4.y.short_name} = 1 + {rel4.coeffs[0].val[0]*1e7:.3f}\\cdot 10^{{{-7}}} {rel4.x.short_name}$'
rel4.plot_fit(ax4, label='lineární fit')
rel4.show_equation(ax4, combined=False, eq_string=eq_string, newline=False)
ax4.ticklabel_format(style='plain', axis='y', useOffset=False)
pr.make_legend(ax4, rel4)
fig4.savefig('uloha19/index_na_tlaku_4.png', dpi=300)

fig5, ax5 = plt.subplots(layout='constrained')
rel5.plot_data(ax5, label='naměřené hodnoty', err=(0,0))
eq_string = f'${rel5.y.short_name} = 1 + {rel5.coeffs[0].val[0]*1e7:.3f}\\cdot 10^{{{-7}}} {rel5.x.short_name}$'
rel5.plot_fit(ax5, label='lineární fit')
rel5.show_equation(ax5, combined=False, eq_string=eq_string, newline=False)
ax5.ticklabel_format(style='plain', axis='y', useOffset=False)
pr.make_legend(ax5, rel5)
fig5.savefig('uloha19/index_na_tlaku_5.png', dpi=300)
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)
plt.close(fig5)

plt.show()

#tau = 2.65072e-7
################################ vlnove delky #####################
#pr.best_print(n)
#pr.best_print(n[-1])

lambda_n = float(lambda_Na.val[0]) * n[-1] / n
lambda_n.set_lname('\\lambda', 'nm')
lambda_Na_0 = lambda_n[0]
lambda_Na_00 = pr.Var(589.45488442254, 0.0001, 'lambda_Na_0', 'nm')
#pr.best_print(lambda_Na_0)
#pr.best_print(lambda_Na_00)
#print(pr.to_table_2(n, lambda_n))
lambda_rel = pr.Rel(n, lambda_n, pr.F.hyperbolic)
lambda_rel.fit(p0=[lambda_Na_0.val[0]], absolute_sigma=True, get_unit=False)
#print(f"Rel lambda: {lambda_rel.chi}")
pr.best_print(lambda_rel.coeffs[0])
figlambda, axlambda = plt.subplots(layout='constrained')
lambda_rel.plot_data(axlambda, err=(0,0))
lambda_rel.plot_fit(axlambda, label='proložená hyperbolická závislost')
lambda_rel.show_equation(axlambda, format='.5f', newline=False)
lambda_rel.make_legend(axlambda)
axlambda.ticklabel_format(style='plain', axis='x', useOffset=False)
figlambda.savefig('uloha19/lambda_na_n.png', dpi=300)
plt.close(figlambda)
plt.show()


########################### teplota #################################
p0 = pr.NonErrorVar(1013.25, 'p0', 'hPa')
gamma = pr.NonErrorVar(3670e-6, '\\gamma', '1/K', fmt='.0e')
gamma6 = pr.NonErrorVar(3670, '10^6 \\cdot \\gamma', '1/K', fmt='.0f')


n15p0_minus1_e6 = 64.328+29498.1/(146-1000/lambda_Na_0.val[0]**2) + 255.4/(41-1000/lambda_Na_0.val[0]**2)
n15p0_minus1 = pr.NonErrorVar(n15p0_minus1_e6/1e6, 'n_{15p0}-1', fmt='.5f')

t = 1/gamma * (n15p0_minus1 * (1 + 15*gamma) / (tau * p0) - 1)
t.set_lname('t', '\\degree C')
pr.best_print(t)


print('All done!')