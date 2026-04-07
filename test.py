import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import praktika as pr

'''
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

df = pd.read_excel('2zs/uloha26/uloha_26.xlsx', sheet_name='List2', skiprows=1, usecols='D:M', nrows=6)
print(df)
print(df.to_latex(index=False))
'''

#####################################################
'''
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
'''

##############################################
'''
n = 20
x = np.linspace(0,10,n)
y = 0.1 * x

data = y + np.random.normal(0,0.1, size=n)
errors = 0.1


plt.figure(4)
plt.plot(x,y,label='model')
#plt.plot(x,data)

plt.errorbar(x,data,errors,fmt='rs',capsize=7,label='experiment')
plt.xlabel(r'x [cm]')
plt.ylabel('A [cm$^{-1}$]')
plt.legend()
'''

##############################################
'''
x = ufloat(12, 0.02859)
print(f'{x:.3uP}')
print(f'{x:.3L}')
print(f'{x:.3uL}')
print(f'{x:.3ueL}')
print(f'{x:.3uS}')
z = ufloat(0.144,0.1)
print(f'{z:.2uL}')
print(f'{z:.2uS}')

y = unp.uarray([1.2345, 2.3456, 3.4567], [0.045, 0.056, 0.067])
print(y)
'''

##############################################
'''
rrr = pr.Var(1.08, 0.01, 'R', '\\Omega')
xxx = rrr - 1
yyy = 2 - rrr
print(xxx.unc)
print(yyy.unc)
'''

##############################################
'''
print('log', unp.log(y))
print('exp', unp.exp(y))
z = unp.log(y)[1]
print(unp.nominal_values(unp.exp(z)))
'''
##############################################
'''
df = pd.DataFrame({
    'ranges': [10, 100, 1000],
    'resolution': [0.01, 0.1, 1],
    'variable_error': [0.5, 0.3, 0.2],
    'constant_error': [0.1, 0.05, 0.02]
})
print(df)
print(df.ranges)
print(df['ranges'])
print(df.iloc[0, :])
print(df.iloc[0])
print(df.iloc[:,0])
print('ff')
print([r for r in df.ranges])
net = 555 < df.ranges.to_numpy()
print(net)
print(df.ranges[net])
print(df.ranges[net].idxmin())
print('multiple indexing')
print(df[['ranges', 'resolution']])
df[['ranges', 'resolution']] = df[['ranges', 'resolution']] * 10
print(df)
'''
###################################################
'''
a = pr.Var(10, 0.03, 'a', 'm')
b = pr.Var(20, 0.04, 'b', 'm')
c = a + b
print(a)
print(b)
print(c)
print(a.unc, a.val, a.err, a.unit)
for val in a.val:
    print('val:', val)
d = pr.Var(a, errors=0.05, short_name='d', unit='m')
print(d)
e = pr.Var(a * 2, errors='1%', short_name='e', unit='m')
print(e)
f = pr.Var(a.val, errors=0.05, short_name='f', unit='m')
print(f)
'''
###########################################################
'''
a = pr.Var([0.1,2.3,4,5.1],'2%','promenna A')
if isinstance(a, pr.Var):
    print(f'{a.short_name} je Var.')
'''
############################################################
'''
xs = pr.Var([1,2,3,4,5,6,7], 0.1, 'x', 'm')
vals = [1.22, 1.21, 1.20, 1.19, 1.21, 1.22, 1.20]
errs = [0.7, 0.8, 0.7, 0.7, 0.6, 0.7, 0.8]
a = pr.Var(vals, 0.7, 'a', 'm')

mean = pr.weighted_mean(a, absolute_sigma=False)
mean_without_err = pr.mean(a.val)
std = pr.std(a.val)
std_mean = std / np.sqrt(len(a.val))
pr.best_print(mean)
pr.best_print(mean_without_err)
print('std:', std)
print('std_mean:', std_mean)
print(mean.chi)

rel = pr.Rel(xs, a, pr.F.const)
rel.fit(absolute_sigma=False)
pr.best_print(rel.coeffs[0])
print(rel.coeffs[0].err)
fig, ax = plt.subplots()
rel.plot_data(ax)
rel.plot_fit(ax)
rel.show_equation(ax)
rel.make_legend(ax)
#plt.close(fig)
plt.show()

from scipy.optimize import curve_fit
# 1. Perform the fit
popt, pcov = curve_fit(pr.F.const, xs.val, a.val, sigma=a.err, absolute_sigma=False)

# 2. Calculate Residuals
residuals = a.val - pr.F.const(xs.val, *popt)

# 3. Calculate Reduced Chi-Square
# This is the "scaling factor" SciPy used
chisq_red = np.sum((residuals / a.err)**2) / (len(xs.val) - len(popt))

# 4. The "Right" Errors (already scaled by pcov)
perr = np.sqrt(np.diag(pcov))

print(f"Scaling factor (Reduced Chi-Square): {chisq_red:.4f}")
print(f"Corrected Slope Error: {perr[0]}")

# chisq_red > 1 => absolute_sigma=False
# chisq_red < 1 => absolute_sigma=True
#'''

####################################################


print('All done!')