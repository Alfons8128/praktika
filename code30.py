import numpy as np
import praktika as pr
import matplotlib.pyplot as plt
import pandas as pd

lambda_na = pr.NonErrorVar(589.3, '\\lambda_\\mathrm{Na}', 'nm')
sigma_x = 0.05

df_tloustka = pr.read_excel('uloha30/uloha30.ods', cells='A3:I7')
df_tloustka.columns = ['misto', 'x1l', 'x1r', 'x1', 'x2l', 'x2r', 'x2', 't', 'sigma_t']

x1l = pr.Var(df_tloustka['x1l'].to_numpy(), sigma_x, '\\mathrm{x_{1l}}', 'dílek')
x1r = pr.Var(df_tloustka['x1r'].to_numpy(), sigma_x, '\\mathrm{x_{1r}}', 'dílek')
x1 = x1r - x1l
x2l = pr.Var(df_tloustka['x2l'].to_numpy(), sigma_x, '\\mathrm{x_{2l}}', 'dílek')
x2r = pr.Var(df_tloustka['x2r'].to_numpy(), sigma_x, '\\mathrm{x_{2r}}', 'dílek')
x2 = x2r - x2l
t = x1 / x2 * lambda_na/2
t.set_lname('\\mathrm{t}')
#pr.best_print(t)

#print(pr.to_table_2(df_tloustka['misto'], x1l, x1r, x1, x2l, x2r, x2, t))

n = pr.NonErrorVar(1.51673, 'n')
t0 = pr.Var(5521623, 90, 'T_0', 'px/m')
sigma_r50_1 = 30.27 * np.sqrt(4 * (90/5521623)**2 + (0.005/30.27)**2)
sigma_r50_2 = 172.38 * np.sqrt(4 * (90/5521623)**2 + (0.09/172.38)**2)
sigma_r100_1 = 60.35 * np.sqrt(4 * (90/5521623)**2 + (0.013/60.35)**2)
sigma_r100_2 = 354.4 * np.sqrt(4 * (90/5521623)**2 + (0.4/354.4)**2)
print('nejistoty',sigma_r50_1, sigma_r50_2, sigma_r100_1, sigma_r100_2)

r50_1 = pr.Var(30.27, sigma_r50_1, 'R_1', 'mm')
r50_2 = pr.Var(-172.38, 0.09, 'R_2', 'mm')
r100_1 = pr.Var(60.35, 0.013, 'R_1', 'mm')
r100_2 = pr.Var(-354.4, 0.4, 'R_2', 'mm')

pr.best_print(r50_1)
pr.best_print(r50_2)
pr.best_print(r100_1)
pr.best_print(r100_2)

d50 = pr.Var(6.5, 0.05, 'd', 'mm')
d100 = pr.Var(4.0, 0.05, 'd', 'mm')
# r50_1 = pr.Var(30.06, 0.01, 'R_1', 'mm')
# r50_2 = pr.Var(-172.0, 0.1, 'R_2', 'mm')
# r100_1 = pr.Var(60.02, 0.01, 'R_1', 'mm')
# r100_2 = pr.Var(-353.30, 0.01, 'R_2', 'mm')

f50 = n/(n-1) * r50_1 * r50_2 / (n*(r50_2 - r50_1) + (n-1)*d50)
f100 = n/(n-1) * r100_1 * r100_2 / (n*(r100_2 - r100_1) + (n-1)*d100)
f50.set_lname('f', 'mm')
f100.set_lname('f', 'mm')

r501 = 30.27
r502 = -172.38
r1001 = 60.35
r1002 = -354.4

s501 = 0.005
s502 = 0.09
s1001 = 0.013
s1002 = 0.4

n=1.51673
d50=6.5
d100=4.0
hranata50 = n*(r502 - r501) + (n-1)*d50
hranata100 = n*(r1002 - r1001) + (n-1)*d100
a50 = r502 * hranata50
b50 = r501 * r502**2 * n
a100 = r1002 * hranata100
b100 = r1001 * r1002**2 * n
sf50 = n / (n-1) * np.sqrt((r502*(n*(r502 - r501) + (n-1)*d50) + n*r501*r502)**2 * s501**2 + (r501*(n*(r502 - r501) + (n-1)*d50) - n*r501*r502)**2 * s502**2 + (r501*r502*(n-1))**2*0.05**2) / (n*(r502 - r501) + (n-1)*d50)**2
sf100 = n / (n-1) * np.sqrt((r1002*(n*(r1002 - r1001) + (n-1)*d100) + n*r1001*r1002)**2 * s1001**2 + (r1001*(n*(r1002 - r1001) + (n-1)*d100) - n*r1001*r1002)**2 * s1002**2 + (r1001*r1002*(n-1))**2*0.05**2) / (n*(r1002 - r1001) + (n-1)*d100)**2

#sf50 = n/(n-1) * np.sqrt((a50 + b50)**2 * s501**2 + (a50 - b50)**2 * s502**2) / hranata50**2
#sf100 = n/(n-1) * np.sqrt((a100 + b100)**2 * s1001**2 + (a100 - b100)**2 * s1002**2) / hranata100**2

print('ruční výpočet nejistot ohniskovych vzdalenosti', sf50, sf100)

pr.best_print(f50)
pr.best_print(f100)
a = pr.Var(3, 0.1, 'a', 'mm')
b = pr.Var(4, 0.1, 'b', 'mm')
print('a+a', a+a, 'spatne 0.1414, spravne 0.2')
print('2a', 2*a)

print('All done!')