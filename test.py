import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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




print('All done!')