import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um

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

x = unp.uarray([1.34, 134.5544, 24.4455, 24.4455, 24.4455], [0.001, 0.0785, 0.025, 0.035, 0.04])
y = unp.uarray([1.34, 134.5544, 24.4455, 24.4455, 24.4455], [0.001, 0.01, 0.025, 0.075, 0.06])
z = x + y
for i in range(len(z)):
    print(i,':', z[i].format('.6uL'))
print(x)
for i in range(len(x)):
    print(i,':', x[i].format('.2ueL'))
y = ufloat(1.2345, 0.044494)
print(y)

x2 = unp.uarray([1.36, 1.54, 1.42, 1.47, 1.43, 1.38], [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
print(x2[0])
print('mean:', np.mean(x2))
print('std:', np.std(unp.nominal_values(x2), ddof=0))
print('std manually:', np.sqrt(np.sum((unp.nominal_values(x2) - np.mean(unp.nominal_values(x2)))**2) / (len(x2))))

print(''.join(str(x) for x in [1,2,'ab','c'] if str(x).isalpha()))


print('All done!')