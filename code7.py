import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

import praktika as pr

df1 = pr.read_excel('uloha5/uloha5.xlsx', 'List1', 'A2:H8')
df1.columns = ['ua', 'dua', 'ia', 'dia', 'uc', 'duc', 'ic', 'dic']

f = 50  # Hz
f = pr.Var(f, 0.0001*f, 'f', 'Hz')
w = 2 * np.pi * f
w.set_lname('\\omega', 's^{-1}')

df2 = pr.read_excel('uloha5/uloha5.xlsx', 'List1', 'A10:H16')
df2.columns = ['uf', 'duf', 'if', 'dif', 'ucf', 'ducf', 'icf', 'dicf']

ua = pr.Var(df1['ua'].to_numpy(), df1['dua'].to_numpy(), 'U_A', 'V')
