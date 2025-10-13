import numpy as np
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt

def read_excel(file_path, sheet_name='List2', cells='A1:Z100'):
    '''Reads an Excel file and returns a pandas DataFrame.'''

    start, end = cells.split(':')
    scol = ''.join(filter(str.isalpha, start))
    ecol = ''.join(filter(str.isalpha, end))
    srow = ''.join(filter(str.isdigit, start))
    erow = ''.join(filter(str.isdigit, end))
    skiprows = int(srow) - 1
    nrows = int(erow) - skiprows
    cols = scol + ':' + ecol

    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows, nrows=nrows, usecols=cols)

    return df

if __name__ == "__main__":
    # Example usage of read_excel function

    df = read_excel('uloha26/uloha_26.xlsx', cells='D2:M7')

    print(df)
    print(df.to_latex(index=False))
