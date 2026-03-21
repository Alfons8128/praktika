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

r50_1 = pr.Var(30.27, 0.005, 'R_1', 'mm')
r50_2 = pr.Var(-172.38, 0.09, 'R_2', 'mm')
r100_1 = pr.Var(60.35, 0.013, 'R_1', 'mm')
r100_2 = pr.Var(-354.4, 0.4, 'R_2', 'mm')

d50 = pr.NonErrorVar(6.5, 'd', 'mm')
d100 = pr.NonErrorVar(4.0, 'd', 'mm')

f50 = n/(n-1) * r50_1 * r50_2 / (n*(r50_2 - r50_1) + (n-1)*d50)
f100 = n/(n-1) * r100_1 * r100_2 / (n*(r100_2 - r100_1) + (n-1)*d100)
f50.set_lname('f', 'mm')
f100.set_lname('f', 'mm')

pr.best_print(f50)
pr.best_print(f100)

print('All done!')