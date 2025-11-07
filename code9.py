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

########################### staticka va charakteristika ##########################
va = pd.read_csv('uloha9/nemec_VA.txt', sep='\s+', header=None)
va.columns = ['i', 'iunit', 'itype', 'u', 'uunit', 'utype']
print(va)
i = pr.Var(va['i'], errors='1%', unit='mA', short_name='I')
u = pr.Var(va['u'], errors='1%', unit='V', short_name='U')
for val in i.unc:
    if val.nominal_value < 2:
        val.std_dev = 0.003 * val.nominal_value + 3*1e-4
        continue
    if val.nominal_value < 20:
        val.std_dev = 0.003 * val.nominal_value + 3*1e-3
        continue
    if val.nominal_value < 200:
        val.std_dev = 0.003 * val.nominal_value + 3*1e-2
        continue

for val in u.unc:
    if val.nominal_value < 0.2:
        val.std_dev = 0.005 * val.nominal_value + 3*1e-5
        continue
    if val.nominal_value < 2:
        val.std_dev = 0.005 * val.nominal_value + 3*1e-4
        continue

print('Table VA:')
pr.to_table(i, u)

########################## odpor na teplote #########################

rt = pd.read_csv('uloha9/nemec_RT.txt', sep='\s+', header=None)
rt.columns = ['pt', 'ptunit', 'pttype', 'rt', 'rtunit', 'rttype']
print(rt)
pt = pr.Var(rt['pt'].to_numpy()[2:], errors='1%', unit='\\Omega', short_name='R_{Pt,100}')
rt = pr.Var(rt['rt'].to_numpy()[2:], errors='1%', unit='k\\Omega', short_name='R_{term}')

for val in pt.unc:
    val.std_dev = 0.002 * val.nominal_value + 5*1e-5

for val in rt.unc:
    if val.nominal_value < 2:
        val.std_dev = 0.0015 * val.nominal_value + 3*1e-7
        continue
    if val.nominal_value < 20:
        val.std_dev = 0.0015 * val.nominal_value + 3*1e-6
        continue
    if val.nominal_value < 200:
        val.std_dev = 0.0015 * val.nominal_value + 3*1e-5
        continue

r0 = pr.Var(100, errors=0, short_name='R_{0,Pt100}', unit='\\Omega')
alpha = pr.Var(0.00385, errors=0, short_name='\\alpha', unit='K^{-1}')
temp = (pt - r0) / (r0 * alpha)
temp.set_lname('T', '°C')

lnr = (1000 * rt).ln()
lnr.set_lname('\\ln R')

oneovert = 1 / temp
oneovert.set_lname('T^{-1}', 'K^{-1}')

print('Table RT:')
pr.to_table(pt, temp, oneovert, rt, lnr)
