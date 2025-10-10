import numpy as np
import matplotlib.pyplot as plt

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

plt.show()



print('All done!')