import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from uncertainties import ufloat, unumpy as unp, umath as um
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
from matplotlib.lines import Line2D
from scipy.stats.distributions import t
from scipy.interpolate import make_interp_spline
import warnings

################# Var class ############################
class Var:
    '''Class defining variables for experimental physics analysis. Atributes are:
    values: scalar/1d-array, errors: scalar/1d-array or str with value in %, short_name: str, unit: str
    
    methods:
    artithmetical operation with error propagation: + - * /
    __get_item__(idx): returns Var instance with selected items
    set_lname(short_name, unit): defines long_name for matplolib axes labels
    ufmt(format description): uses scalar ufmt function to round values to apropriate decimal digits
    '''
    
    def __init__(self, values: np.array, errors=0, short_name='', unit=None):
        # input is Var type
        if isinstance(values, Var):
            self.unc = np.atleast_1d(values.unc)

        # errors in percents
        elif isinstance(errors, str):
            if errors.endswith('%'):
                errors = float(errors[:-1]) / 100.0
            else:
                errors = float(errors)
            errors = np.abs(values) * errors
            self.unc = np.atleast_1d(unp.uarray(values, errors))

        else:
            self.unc = np.atleast_1d(unp.uarray(values, errors))

        self.val = unp.nominal_values(self.unc)
        self.err = unp.std_devs(self.unc)
        self.short_name = short_name
        self.unit = unit
        self.long_name = f'${self.short_name}\\, (\\mathrm{{{self.unit}}})$' if self.unit else f'${self.short_name}$'

    def set_lname(self, short_name, unit=None):
        self.short_name = short_name
        self.unit = unit
        self.long_name = f'${self.short_name}\\, (\\mathrm{{{self.unit}}})$' if self.unit else f'${self.short_name}$'

    def __str__(self):
        if self.unit:
            return f'{self.short_name} = ({", ".join(scalar_ufmt(x, 'P') for x in self.unc)}) {self.unit}'
        return f'{self.short_name} = ({", ".join(scalar_ufmt(x, 'P') for x in self.unc)})'
    
    def latex(self):
        if self.unit:
            return f'{self.short_name} = ({", ".join(scalar_ufmt(x, "L") for x in self.unc)}) \\cdot {self.unit}'
        return f'{self.short_name} = ({", ".join(scalar_ufmt(x, "L") for x in self.unc)})'
        

    def __add__(self, other):
        if isinstance(other, Var):
            new_unc = self.unc + other.unc
        else:
            new_unc = self.unc + other

        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), self.short_name, self.unit)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Var):
            new_unc = self.unc - other.unc
        else:
            new_unc = self.unc - other

        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), self.short_name, self.unit)

    def __rsub__(self, other):
        result_unc = other - self.unc
        return Var(unp.nominal_values(result_unc), unp.std_devs(result_unc), self.short_name, self.unit)
    
    def __mul__(self, other):
        if isinstance(other, Var):
            new_unc = self.unc * other.unc
        else:
            new_unc = self.unc * other

        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), self.short_name, self.unit)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Var):
            new_unc = self.unc / other.unc
        else:
            new_unc = self.unc / other

        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), self.short_name, self.unit)
    
    def __rtruediv__(self, other):
        result_unc = other / self.unc
        return Var(unp.nominal_values(result_unc), unp.std_devs(result_unc), self.short_name, self.unit)
    
    def __pow__(self, power):
        new_unc = self.unc ** power
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), self.short_name, self.unit)
    
    def ln(self):
        new_unc = unp.log(self.unc)
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), f'\\ln {self.short_name}', self.unit)
    
    def exp(self):
        new_unc = unp.exp(self.unc)
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), f'\\exp {self.short_name}', self.unit)
    
    def sqrt(self):
        new_unc = unp.sqrt(self.unc)
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), f'\\sqrt{{{self.short_name}}}', self.unit)
    
    def __setitem__(self, idx, value):
        new_unc = self.unc
        if isinstance(value, Var):
            new_unc[idx] = value.unc
        elif isinstance(value, ufloat) or isinstance(value, unp.uarray):
            new_unc[idx] = value
        else:
            new_unc[idx] = unp.uarray(value, 0)
        self.unc = new_unc
        self.val = unp.nominal_values(self.unc)
        self.err = unp.std_devs(self.unc)

    
    def __getitem__(self, idx):
        new_unc = self.unc[idx]
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), self.short_name, self.unit)
    
    def ufmt(self, apx='L'):
        '''Formats all values in the Var instance using ufmt function.'''
        return [scalar_ufmt(x, apx=apx) for x in self.unc]

############### Function class #########################
class F:
    '''A collection of common fitting functions.'''
    def const(x, constant):
        return constant * np.ones_like(x)
    
    def direct(x, slope):
        return slope * x
    
    def linear(x, intercept, slope):
        return slope * x + intercept
    
    def polynomial(x, *coeffs):
        '''coeffs are in increasing order, i.e., coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...'''
        p = Polynomial(coeffs)
        return p(x)

    def power(x, amplitude, power):
        return amplitude * x ** power

    def exp(x, amplitude, xscale):
        return amplitude * np.exp(xscale * x)

    def log(x, amplitude, xscale):
        return amplitude * np.log(xscale * x)
    
    def resonance(x, d):
        return (d * d) / (d*d + (x - 1/x)**2)

############### Relation class #########################
class Rel:
    '''This class provides a relation between two variables (from class Var):
    independent x and dependent y defined by specific function. That function
    is then used for fitting the data: y = func(x).
    Atributes: independent Var, dependent Var, function, coefficients, covariant matrix.
    Methods: fit, plot_data, plot_function, show_equation.'''

    def __init__(self, x: Var, y: Var, function: callable = None):
        self.x = x
        self.y = y
        self.func = function

    def fit(self, p0: list = None, get_unit: bool = True):
        '''Fits y over x (both Var instances) using the provided function.
        Adds these attributes to Relation: coefficients (.coeffs), covariant matrix (.cov).'''

        if self.func is None:
            raise ValueError('No fitting function defined for the Relation.')

        coeff_values, cov_matrix  = curve_fit(self.func, self.x.val, self.y.val, sigma=self.y.err, absolute_sigma=True, p0=p0)
        #alpha = 0.05  # 95% confidence interval
        #t_val = t.ppf(1.0-alpha/2, max(0, len(x.val)-len(coeff_values)))
        coeff_errors = np.sqrt(np.diag(cov_matrix)) #* np.abs(t_val)
        fitted_coeffs = [Var(coeff_values[i],coeff_errors[i], short_name=f'fit_coefficient_{i}') for i in range(len(coeff_values))]

        self.coeffs = fitted_coeffs
        self.cov = cov_matrix

        if get_unit:
            match self.func:
                case F.const:
                    self.coeffs[0].unit = self.y.unit
                case F.direct:
                    self.coeffs[0].unit = f'{self.y.unit}/{self.x.unit}'
                case F.linear:
                    self.coeffs[0].unit = self.y.unit
                    self.coeffs[1].unit = f'{self.y.unit}/{self.x.unit}'
                case _:
                    warnings.warn(f'Unit calculation not implemeted for {self.func.__name__} function, \
                                  no units assigned to fitted coefficients.')
    
    def plot_data(self, ax, err: tuple = (1,1), connect=False, smooth=False, label='naměřené hodnoty', 
                  scale=1, marker='s', color='black', zorder=None):
        match err:
            case (1, 1): # x_error_bar and y_error_bar
                    ax.errorbar(self.x.val, self.y.val, xerr=self.x.err, yerr=self.y.err, marker=marker,
                                linewidth=1, color=color, markersize=5*scale, capsize=3, label=label, linestyle='', zorder=zorder)
            case (1, 0): # just x_error_bar
                    ax.errorbar(self.x.val, self.y.val, xerr=self.x.err, marker=marker, linewidth=1, 
                                color=color, markersize=5*scale, capsize=3, label=label, linestyle='', zorder=zorder)
            case (0, 1): # just y_error_bar
                    ax.errorbar(self.x.val, self.y.val, yerr=self.y.err, marker=marker, linewidth=1, 
                                color=color, markersize=5*scale, capsize=3, label=label, linestyle='', zorder=zorder)
            case (0, 0): # none of the errorbars
                ax.scatter(self.x.val, self.y.val, marker=marker, s=25*scale, color=color, linewidth=1, label=label, zorder=zorder)

        if connect:
            ax.plot(self.x.val, self.y.val, color='black')

        if smooth:
            # Create a smooth line using spline interpolation
            x_smooth = np.linspace(min(self.x.val), max(self.x.val), 300)
            spline = make_interp_spline(self.x.val, self.y.val, k=smooth)  # Cubic spline
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color='black', linestyle='--')

        ax.set_xlabel(self.x.long_name)
        ax.set_ylabel(self.y.long_name)
        #ax.legend()


    def plot_fit(self, ax, label='fitovaná přímka', linestyle=':', color='black', linewidth=1.5):
        if self.func is None:
            raise ValueError('No fitting function defined for the Relation.')
        if not hasattr(self, 'coeffs'):
            raise ValueError('No fitted coefficients found. Please run the fit() method first.')

        x = np.linspace(min(self.x.val), max(self.x.val), 200)
        ax.plot(x, self.func(x, *self.coeffs.val), linestyle=linestyle, color=color, linewidth=linewidth, label=label)
        #ax.legend()


    def show_equation(self, ax, label=None):
        match self.func:
            case F.const:
                eq_string = f'${self.y.short_name} = {self.coeffs.val[0]:.3f}$'
            case F.direct:
                eq_string = f'${self.y.short_name} = {self.coeffs.val[0]:.3f} \\cdot {self.x.short_name}$'
            case F.linear:
                eq_string = f'${self.y.short_name} = {self.coeffs.val[0]:.3f} \\cdot {self.x.short_name} +' 
                f'{self.coeffs.val[0]:.3f}$'

        combined_handle = Line2D([], [], color='black', marker='s', linestyle=':', label=eq_string)
        ax.legend(handles=[combined_handle])

############### Measuring tool uncertainty class #######
class MeasureUnc:
    '''Class for storing uncertainty values of certaint measuring tool.'''
    #def __init__(self, err_type: tuple, unit: str, data: pd.DataFrame):
    def __init__(self, err_type: str, unit: str, data: pd.DataFrame):
        '''err_type: ('percent (of measured value)', 'digit/range')
           data: columns are range, resolution, variable error and constant error'''
        self.err_type = err_type
        self.unit = unit
        data.columns = ['ranges', 'resolution', 'variable_error', 'constant_error']
        self.data = data
    
    def convert_units(self, to_unit: str):
        from_unit = self.unit
        if from_unit == to_unit:
            return self
        # implement conversion factors here
        Prefix = {
            'da': 1e1,
            'h': 1e2,
            'k': 1e3,
            'M': 1e6,
            'G': 1e9,
            'T': 1e12,
            'P': 1e15,
            'd': 1e-1,
            'c': 1e-2,
            'm': 1e-3,
            'u': 1e-6,
            'n': 1e-9,
            'p': 1e-12,
            'f': 1e-15
        }
        
        factors = []
        for unit in [from_unit, to_unit]:
            factor = 1.0
            for prefix in Prefix.keys():
                if unit.startswith(prefix):
                    factor = Prefix[prefix]
            if factor == 1.0:
                warnings.warn(f'No prefix found for unit {unit}, assuming factor 1.')
            #else:
            #    raise ValueError(f'Unit conversion for {unit} not implemented.')
            
            factors.append(factor)

        from_factor = factors[0]
        to_factor = factors[1]
        conv_factor = from_factor / to_factor
        self.data[['ranges', 'resolution']] = self.data[['ranges', 'resolution']] * conv_factor

        return self
    
    def set_uncertainty(self, var: Var):
        '''Adds uncertainty to the provided Var instance based on the measuring tool uncertainty.'''
        self = self.convert_units(var.unit)
        errors = np.zeros_like(var.val)

        #if self.err_type[0] == 'percent':
        #    var_err_idx = 2
        if self.err_type == 'digit':
            const_base = self.data.resolution.to_numpy()
        if self.err_type == 'range':
            const_base = self.data.ranges.to_numpy()
            self.data.constant_error = self.data.constant_error / 100 # convert from percent to fraction
        
        for i, value in enumerate(var.val):
            net = value < self.data.ranges
            row_idx = self.data.ranges[net].idxmin() # first range larger than value

            errors[i] += np.abs(value) * (self.data.variable_error[row_idx] / 100.0) # variable percent error
            errors[i] += const_base[row_idx] * self.data.constant_error[row_idx] # (digits)resolution/range * constant error

        return Var(var.val, errors, var.short_name, var.unit)


########################################################
########### other useful functions #####################
def read_excel(file_path, sheet_name='List1', cells='A1:Z100', header = 0):
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
########################################################
def excel_to_latex(file_path, cells='A1:Z100', sheet_name='List1', header = 0, format='.2f'):
    '''Reads an Excel file and returns a LaTeX formatted table as a string.'''
    df = read_excel(file_path, sheet_name=sheet_name, cells=cells, header=header)
    cols = df.columns.to_list()
    nrows, ncols = df.shape
    print(df.shape)
    string = f'\n' + \
    '\\begin{table}[!htbp]' + '\n' + \
    '\\centering' + '\n' + \
    '\\caption{CAPTION}' + '\n' + \
    '\\begin{tabular}{' + 'c'*ncols + '}' + '\n' + \
    ' \\toprule' + '\n' + \
    ' ' +' & '.join(cols) + ' \\\\' + '\n' + \
    ' \\midrule' + '\n'
    for i in range(nrows):
        string += ' $' + '$ & $'.join(f'{df.iloc[i,j]:{format}}' for j in range(ncols)) + '$ \\\\' + '\n'
    string += ' \\bottomrule' + '\n'
    string += '\\end{tabular}' + '\n'
    string += '\\label{tab:LABEL}' + '\n'
    string += '\\end{table}' + '\n'

    return string

########################################################
def scalar_ufmt(x, apx='L'):
    '''Work only for scalar values!
    
    Rounds value and error of an ufloat number to appropriate significant figures.
    Returns formatted string. If first significant figure of error is 1 (k <= 1.9), uses two significant digits.
    Appendix (apx) sets formatting options: e for scientific notation,
    L for LaTeX.'''

    if x.std_dev == 0:
        return f'{x:.2u{apx}}'
    
    # assume error = k * 10^mag
    mag = int(np.floor(np.log10(x.std_dev)))
    k = x.std_dev / (10 ** mag)
    sig_fig = 2 if k <= 1.9 else 1

    return f'{x:.{sig_fig}u{apx}}'

########################################################
def nice_print(Var, apx='P'):
    '''Prints ufloat number with nicely formatted value and error.'''
    if Var.unit:
        print(f'{Var.short_name} = ({", ".join(scalar_ufmt(x, apx) for x in Var.unc)}) \\times {Var.unit}')
    else:
        print(f'{Var.short_name} = ({", ".join(scalar_ufmt(x, apx) for x in Var.unc)})')

########################################################
def to_table(*args, apx='L'):
    '''Converts Var instances to a formatted LaTeX table. Uses ufmt for formatting, 
    defaultly writes uncertainties in LaTeX format.'''
    # df = pd.DataFrame()
    # for var in args:
    #     df[var.long_name] = [scalar_ufmt(x, apx=apx) for x in var.unc]
    # return df.to_latex(index=False, column_format=len(args)*'c')

    print('\\begin{table}[!htbp]')
    print('\\centering')
    print('\\caption{CAPTION}')
    print('\\begin{tabular}{' + 'c'*len(args) + '}', sep='')
    print(' \\toprule')
    print(' ' +' & '.join(arg.long_name for arg in args) + ' \\\\')
    print(' \\midrule')
    for i in range(len(args[0].unc)):
        print(' $' + '$ & $'.join(scalar_ufmt(arg.unc[i], apx=apx) for arg in args) + '$ \\\\')
    print(' \\bottomrule')
    print('\\end{tabular}')
    print('\\label{tab:LABEL}')
    print('\\end{table}')

########################################################
def fit_curve(x: Var, y: Var, func=F.linear, p0: list =None):
    '''Fits y over x (both Var instances) using the provided function (default linear).
    Returns the fitted coefficients as uarray.'''
    #alpha = 0.05  # 95% confidence interval
    coeff_values, cov_matrix  = curve_fit(func, x.val, y.val, sigma=y.err, absolute_sigma=True, p0=p0)
    #t_val = t.ppf(1.0-alpha/2, max(0, len(x.val)-len(coeff_values)))
    coeff_errors = np.sqrt(np.diag(cov_matrix)) #* np.abs(t_val)
    fitted_coeffs = Var([*coeff_values],[*coeff_errors], short_name='fit_coeffs')

    return fitted_coeffs

########################################################
def plot(fig, ax, x, y, fit_coeffs=None, func=F.linear, xerr=True, yerr=True, fit_curve=True, 
         data_label='naměřené hodnoty', fit_label='teoretická závislost', equation=False):

    '''!!!NOT SUPPORTED ANYMORE!!! 
    
    Plots data y over x (both Var instances) with optional error bars, fit line and equation.
    Optionally show the plot.'''

    if fit_curve:
        ax.plot(x.val, func(x.val, *fit_coeffs), 'k:', linewidth=1.5, label=fit_label)

    match (xerr, yerr):
        case (True, True):
                ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err, fmt='ks', linewidth=1, markersize=5, capsize=3, label=data_label)
        case (True, False):
                ax.errorbar(x.val, y.val, xerr=x.err, fmt='ks', linewidth=1, markersize=5, capsize=3, label=data_label)
        case (False, True):
                ax.errorbar(x.val, y.val, yerr=y.err, fmt='ks', linewidth=1, markersize=5, capsize=3, label=data_label)
        case (False, False):
            ax.scatter(x.val, y.val, marker='s', s=25, color='black', linewidth=1, label=data_label)
    
    if equation:
        eq_string = f'${y.short_name} = {fit_coeffs[1]:.3f} \\cdot {x.short_name} + {fit_coeffs[0]:.3f}$'
        combined_handle = Line2D([], [], color='black', marker='s', linestyle=':', label=eq_string)
        ax.legend(handles=[combined_handle])
    else:
        ax.legend()
    
    ax.set_xlabel(x.long_name)
    ax.set_ylabel(y.long_name)

########################################################
def plot_data(ax, x: Var, y: Var, err: tuple = (1,1), label='naměřené hodnoty', scale=1, marker='s', color='black'):
    match err:
        case (1, 1): # x_error_bar and y_error_bar
                ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err, color=color, linewidth=1, 
                            marker=marker, markersize=5*scale, capsize=3, label=label)
        case (1, 0): # just x_error_bar
                ax.errorbar(x.val, y.val, xerr=x.err, color=color, linewidth=1, 
                            marker=marker, markersize=5*scale, capsize=3, label=label)
        case (0, 1): # just y_error_bar
                ax.errorbar(x.val, y.val, yerr=y.err, color=color, linewidth=1, 
                            marker=marker, markersize=5*scale, capsize=3, label=label)
        case (0, 0): # none of the errorbars
            ax.scatter(x.val, y.val, marker=marker, s=25*scale, color=color, linewidth=1, label=label)

    ax.set_xlabel(x.long_name)
    ax.set_ylabel(y.long_name)
    ax.legend()

########################################################
def plot_fit(ax, x: Var, coeffs: Var, func=F.linear, label='fitovaná přímka', linestyle=':', color='black'):
    ax.plot(x.val, func(x.val, *coeffs.val), linestyle=linestyle, color=color, linewidth=1.5, label=label)

########################################################
def show_equation(ax, rel: Rel, label=None):
    pass

########################################################
################## testing #############################
if __name__ == "__main__":

########################################################
    # read_excel
    '''
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
    '''
########################################################
    # fit and plot random data
    '''
    np.random.seed(0)
    x = np.linspace(0, 10, 30)
    y = 3*x**2 + 2*x + 1 + np.random.randn(30)/1000
    dy = 0.5 * np.ones_like(x)
    dx = 0.1 * np.ones_like(x)
    x = Var(x, dx, 'artificial x', 'm')
    y = Var(y, dy, 'artificial y', 'J')
    #print('x:', x)
    #print('y:', y)
    rel = Rel(x, y, F.linear)
    rel.fit()
    print('Fitted coeffs:', [r.latex() for r in rel.coeffs])

    fit, cov = curve_fit(F.linear, x.val, y.val, sigma=dy, absolute_sigma=True)
    inter = fit[0]
    slp = fit[1]
    di = np.sqrt(cov[0][0])
    ds = np.sqrt(cov[1][1])

    fig, ax = plt.subplots()
    ax.errorbar(x.val, y.val, yerr=dy, xerr=dx, fmt='rs', lw=1, ms=3, label='Experiment')
    ax.plot(x.val, F.linear(x.val, *fit), 'b--', label='Fit')
    eq_string = f'${y.short_name} = {fit[1]:.3f} \\cdot {x.short_name} + {fit[0]:.3f}$'
    ax.text(0.4, 0.6, eq_string, transform=ax.transAxes)
    
    #plt.close('all')
    plot_data(ax, x, y)
    plt.show()
    '''
########################################################
    # scalar_ufmt
    '''
    print(scalar_ufmt(ufloat(4.965,0.08)))
    print(scalar_ufmt(ufloat(5.334,0.00134),apx=''))
    print(scalar_ufmt(ufloat(4.222,0.00190)))
    print(scalar_ufmt(ufloat(149.5,56.6)))
    print(scalar_ufmt(ufloat(0.001495,0.000566), 'eL'))
    print(scalar_ufmt(ufloat(14.95,0.566), 'L'))
    '''
########################################################
    # polynomial fit
    '''
    cff = fit_curve(x, y, func=F.polynomial, p0=[1,2,3])
    print('Fitted coeffs:', cff)
    '''
########################################################
    # scalar Var
    '''
    hhh = unp.uarray(1.02,0.03)
    print('hhh:', hhh)
    print(type(hhh))
    print(np.shape(hhh))
    #print('hhhh:', hhh.std_dev)
    print('hhh ufmt:', unp.std_devs(hhh))
    # for x in hhh:
    #     print(x) # will not work, 0-d array is not iterable

    fff = unp.uarray([1.02], [0.03])
    print('fff:', fff)
    print(type(fff))
    print(np.shape(fff))
    print('fff ufmt:', unp.std_devs(fff))
    for x in fff:
        print(x)

    ggg = Var(1.02, 0.03)
    print('type ggg:', type(ggg))
    print('ggg val:', ggg.val)
    print('ggg err:', ggg.err)
    print('ggg:', ggg)
    print(np.shape(ggg.unc))
    print(ggg.unc, ggg.val, ggg.err)
    print('ggg ufmt:', unp.std_devs(ggg.unc))

    print('shape', np.shape(1.02))
    if np.shape(1.02) == ():
        print('scalar')

    '''
########################################################
    # Relation fit and plot
    '''
    fig, ax = plt.subplots()
    np.random.seed(0)
    x = np.linspace(0, 10, 30)
    y = 3*x**2 + 2*x + 1 + np.random.randn(30)/1000
    dy = 0.5 * np.ones_like(x)
    dx = 0.1 * np.ones_like(x)
    x = Var(x, dx, 'artificial x', 'm')
    y = Var(y, dy, 'artificial y', 'J')

    r = Rel(x, y, F.polynomial)
    r.fit([3,1,1])
    r.plot_data(ax)
    r.plot_fit(ax)
    plt.show()
    '''

########################################################
    # type and isinstance tests
    '''
    a = Var(1.0, 0.1, 'a', 'm')
    b = ufloat(2.0, 0.4)
    print(type(a))
    print(isinstance(a, Var))
    print(np.shape(a))
    if type(a) == Var:
        print('a has Var type')
    if isinstance(a, Var):
        print('a is Var instance')
    #if isinstance(b, unp.ufloat): # does not work
    #    print('b is ufloat instance')

    c = unp.uarray(1.2, 0.03)
    c = unp.uarray([1.2],[0.03])
    print(np.shape(c))
    print(c)
    d = 3 * a
    e = Var(3 * a)
    print(d)
    print(e)
    print(d.latex())

    '''
########################################################
    # excel_to_latex
    '''
    latex_table = excel_to_latex('2zs/uloha8/uloha8.xlsx', sheet_name='List1', cells='A2:C7', format='.3f')
    print(latex_table)
    '''
########################################################

    print('All done!')
