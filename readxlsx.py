import numpy as np
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    # Example usage of read_excel function

    df = read_excel('uloha8/uloha8.xlsx', sheet_name='List1', cells='A2:E7')

    print(df)
    print(df.to_latex(index=False))
