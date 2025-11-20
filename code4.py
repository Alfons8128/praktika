import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

import praktika as pr

df1 = pr.read_excel('uloha4/uloha4.xlsx', cells='B3:B12')

df2 = pr.read_excel('uloha4/uloha4.xlsx', cells='B3:B12')

df3 = pr.read_excel('uloha4/uloha4.xlsx', cells='B3:B12')


df4 = pr.read_excel('uloha4/uloha4.xlsx', cells='B3:B12')

df5 = pr.read_excel('uloha4/uloha4.xlsx', cells='B3:B12')


