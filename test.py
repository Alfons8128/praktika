import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat, unumpy as unp

x = np.linspace(0, 5, 100)
y = x**3

cmap = plt.get_cmap('viridis')

plt.scatter(x, y, c=cmap(x), label='y = $x^3$')
plt.grid()
plt.legend()

plt.xscale('log')
plt.yscale('log')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Scatter plot of y = x^3')

#plt.show()
plt.close()

df = pd.read_excel('uloha26/uloha_26.xlsx', sheet_name='List2', skiprows=1, usecols='D:M', nrows=6)
print(df)
print(df.to_latex(index=False))

x = unp.uarray([1.34, 134.5544, 24.4455], [0.001, 0.0785, 0.02])
print(x)


print('All done!')