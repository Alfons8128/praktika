import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import praktika as pr

lambda_hg = pr.NonErrorVar(546.074, short_name='\\lambda_{\\mathrm{Hg}}', unit='nm')

dfkalib1 = pr.read_excel('uloha8/uloha8.ods', cells='A3:C14')
dfkalib2 = pr.read_excel('uloha8/uloha8.ods', cells='A25:C36')

dfmichelson = pr.read_excel('uloha8/uloha8.ods', cells='E3:G14')
dfmichelson.columns = ['k', 'y (mm)', 'delta y (mm)']

dffabryperot = pr.read_excel('uloha8/uloha8.ods', cells='E25:G36')
dffabryperot.columns = ['k (splynutí)', 'l (mm)', 'delta l (mm)']
#print(dffabryperot)

sigma_y = 0.01
sigma_deltay = sigma_y*np.sqrt(2)
k1 = pr.NonErrorVar(dfkalib1['k'], short_name='k')
y1 = pr.Var(dfkalib1['y (mm)'], sigma_y, short_name='y', unit='mm')
deltay1 = pr.Var(dfkalib1['delta y (mm)'], sigma_deltay, short_name='\\Delta y', unit='mm')

k2 = pr.NonErrorVar(dfkalib2['k'], short_name='k')
y2 = pr.Var(dfkalib2['y (mm)'], sigma_y, short_name='y', unit='mm')
deltay2 = pr.Var(dfkalib2['delta y (mm)'], sigma_deltay, short_name='\\Delta y', unit='mm')

kmichelson = pr.NonErrorVar(dfmichelson['k'], short_name='k')
ymichelson = pr.Var(dfmichelson['y (mm)'], sigma_y, short_name='y', unit='mm')
deltaymichelson = pr.Var(dfmichelson['delta y (mm)'], sigma_deltay, short_name='\\Delta y', unit='mm')

kfabryperot = pr.NonErrorVar(dffabryperot['k (splynutí)'], short_name='\\mathrm{k}')
yfabryperot = pr.Var(dffabryperot['l (mm)'], sigma_y, short_name='l', unit='mm')
deltayfabryperot = pr.Var(dffabryperot['delta l (mm)'], sigma_deltay, short_name='\\Delta l', unit='mm')

figkal, axkal = plt.subplots()
kalib1 = pr.Rel(k1, deltay1, pr.F.direct, color='black', shape='s')
kalib2 = pr.Rel(k2, deltay2, pr.F.direct, color='green', shape='o')
kalib1.fit()
for c in kalib1.coeffs:
    print(c)
kalib2.fit()
for c in kalib2.coeffs:
    print(c)

kalib1.plot_data(axkal, label='Kalibrace 1', err=(0,1))
kalib2.plot_data(axkal, label='Kalibrace 2', err=(0,1))

kalib1.plot_fit(axkal, label='Lineární fit kalibrace 1')
kalib2.plot_fit(axkal, label='Lineární fit kalibrace 2')

kalib1.show_equation(axkal, combined=False, format='.6f')
kalib2.show_equation(axkal, combined=False, format='.6f')
#axkal.legend(handles=[*kalib1.handles, *kalib2.handles])
pr.make_legend(axkal, kalib1, kalib2)

figkal.savefig('uloha8/kalibrace.png')
plt.close(figkal)
plt.show()

#print(pr.excel_to_latex_2('uloha8/uloha8.ods', cells='A55:E66', format='.2f'))
#print(pr.to_table_2(k1, y1, deltay1, y2, deltay2))
a1 = kalib1.coeffs[0]
a2 = kalib2.coeffs[0]
p1 = lambda_hg/1e6/(2*a1)
p2 = lambda_hg/1e6/(2*a2)
p = (p1 + p2)/2
pr.best_print(a1)
pr.best_print(a2)
pr.best_print(p1)
pr.best_print(p2)
pr.best_print(p)
pr.best_print((a1 + a2)/2)

michelson = pr.Rel(kmichelson, deltaymichelson, pr.F.direct)
michelson.fit()
figmich, axmich = plt.subplots()
michelson.plot_data(axmich, err=(0,1))
michelson.plot_fit(axmich)
michelson.show_equation(axmich, combined=False, format='.6f')
pr.make_legend(axmich, michelson)

figmich.savefig('uloha8/michelson.png')
plt.close(figmich)
plt.show()

#print(pr.to_table_2(kmichelson, ymichelson, deltaymichelson))

b = michelson.coeffs[0]
b.set_lname('b', unit='mm')
lambda_HeNe = 2 * p * b * 1e6
lambda_HeNe.set_lname('\\lambda_{\\mathrm{HeNe}}', unit='nm')
pr.best_print(b)
pr.best_print(lambda_HeNe)

fabryperot = pr.Rel(kfabryperot, deltayfabryperot, pr.F.direct)
fabryperot.fit()
for c in fabryperot.coeffs:
    pr.best_print(c)
figfp, axfp = plt.subplots()
fabryperot.plot_data(axfp, err=(0,0))
fabryperot.plot_fit(axfp)
fabryperot.show_equation(axfp, combined=False, format='.6f')
pr.make_legend(axfp, fabryperot)
figfp.savefig('uloha8/fabryperot.png')

#plt.close(figfp)
plt.show()
c = fabryperot.coeffs[0]
c.set_lname('c', unit='mm')
pr.best_print(c)
lambda_na = pr.NonErrorVar(589.3, short_name='\\lambda_{\\mathrm{Na}}', unit='nm')
delta_lambda = lambda_na**2 / (2 * c * 1e6)
delta_lambda.set_lname('\\Delta \\lambda', unit='nm')
pr.best_print(delta_lambda)

print(pr.to_table_2(kfabryperot, yfabryperot, deltayfabryperot))

print('All done.')
