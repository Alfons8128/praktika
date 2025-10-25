import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import praktika as pr
from praktika import Var
from functools import partial
from matplotlib.lines import Line2D

df = pr.read_excel('uloha27/uloha27.xlsx', sheet_name='List1', cells='A2:D13')
df.columns = ['u0','ur','i0','c']

u0 = Var(df['u0'].to_numpy(), errors='1%', short_name='U_0', unit='V')
ur = Var(df['ur'].to_numpy(), errors='1%', short_name='U_r', unit='mV')
R = Var(100, short_name='R', unit='\\Omega')
w = Var(2 * np.pi * 100000, short_name='\\omega', unit='s^{-1}')
i0 = ur / R
i0.set_lname('I_0', 'mA')
c = i0 / 1000 / (w * u0) * 1e12
c.set_lname('C', 'pF')

print('Done table:')
pr.to_table(u0, ur, i0, c)


coeffs = pr.fit_curve(u0, c, pr.F.const)
print(coeffs)


df2 = pr.read_excel('uloha27/uloha27.xlsx', sheet_name='List1', cells='A21:C36')
df2.columns = ['d','d-1','c',]