import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_excel(file_path, sheet_name='List2', cells='A1:Z100', header = 0):
    '''Reads an Excel file and returns a pandas DataFrame.
    Defautly, header on the first line.'''

    start, end = cells.split(':')
    scol = ''.join(filter(str.isalpha, start))
    ecol = ''.join(filter(str.isalpha, end))
    cols = scol + ':' + ecol

    srow = int(''.join(filter(str.isdigit, start)))
    erow = int(''.join(filter(str.isdigit, end)))
    skiprows = srow - 1
    nrows = erow - srow    # first is already loaded as header, now need to load nrows rows

    if header == None:
        nrows += 1  # if no header, load one more row


    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows, nrows=nrows, usecols=cols, header=header)

    return df

def linear_f(x, a, b):
    return a * x + b

def power_f(x, a, p):
    return a * x ** p

def exp_f(x, a, b):
    return a * np.exp(b * x)

def log_f(x, a, b):
    return a * np.log(b * x)

if __name__ == "__main__":
    # Example usage of read_excel function

    df = read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='A2:E7')

    print(df)
    print(df.to_latex(index=False))
    for i in range(len(df.iloc[:])):
        df.iloc[i, 1], df.iloc[i, 2] = min(df.iloc[i, [1, 2]]), max(df.iloc[i, [1, 2]])
    
    print(df)

    print(df.iloc[:,[0,3,4]])
    raw_c = df.iloc[:,3].to_numpy()
    c = unp.uarray(raw_c, 0.25)
    raw_f = df.iloc[:,0].to_numpy()
    f = unp.uarray(raw_f, 0.1)
    w = df.iloc[:,4].to_numpy()
    print(c)
    print(f)

    np.random.seed(0)
    x = np.linspace(0, 10, 30)
    y = 3*x**2 + 2*x + 1 + np.random.randn(30)
    dy = 5 * np.ones_like(x)
    dx = 0.7 * np.ones_like(x)

    fit, cov = curve_fit(linear_f, x, y, sigma=dy, absolute_sigma=True)
    inter = fit[0]
    slp = fit[1]
    di = np.sqrt(cov[0][0])
    ds = np.sqrt(cov[1][1])

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=dy, xerr=dx, fmt='rs', lw=1, ms=3, label='Experiment')
    ax.plot(x, linear_f(x, *fit), 'b--', label='Fit')
    
    plt.close('all')



    

