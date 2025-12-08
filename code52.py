import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

def langmuir(U, a, b):
    #if U < b:
    #    return 0
    return a * np.sqrt(U - b)**3

b = - 0.4
a = 4/15 * 2/ 2.132
U = np.linspace(-5, 4, 1000)

#U2 = np.linspace(-5, b-0.01, 1000)
#I2 = np.zeros_like(U2)
u0 = [0, 3.5]
i0 = [0.08, 2]
popt, pcov = curve_fit(langmuir, u0, i0, p0=[a, b])
print('Fitted parameters:', popt)

I = np.array([langmuir(u, *popt) if u >= popt[1] else 0 for u in U])

fig, ax = plt.subplots()
ax.plot(U, I, label='V-A charakteristika vakuové diody')
#ax.plot(U2, I2, color='blue')
ax.set_xlabel('U (V)')
ax.set_ylabel('I ($10^{-2}$ A)')

ax.set_xlim(-5, 5)
ax.set_ylim(-4, 4)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
print('xlim:', xmin, xmax, 'ylim:', ymin, ymax)
gh = 0.85

ax.axhline(2, color='black', linestyle=':', xmin=-5, xmax=0.85)#(3.5 - xmin) / (xmax - xmin))
ax.axhline(0.08, color='black', linestyle=':', xmin=-5, xmax=0.5)
ax.axvline(0, color='black', linestyle=':', ymin=-4, ymax=4.08/8)
ax.axvline(3.5, color='black', linestyle=':', ymin=-4, ymax=6/8)
ax.scatter(u0, i0, color='blue', s=15, zorder=5, label='odhadované hodnoty')

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

ax.grid(which='major', linewidth=1)
ax.grid(which='minor', linewidth=0.5, linestyle='--')
ax.legend()
fig.savefig('uloha5/fig3.png')

plt.show()