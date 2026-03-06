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
    
    def __init__(self, values: float | int, errors: int | str = 0, short_name: str ='', unit: str = None):
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
    
    def weighted_mean(self) -> 'Var':
        '''Calculates the weighted mean of the values with weights as the inverse of variances.
        Returns the mean with uncertainty as a Var instance and adds it as an attribute .mean to the Var instance.'''
        weights = 1 / (self.err ** 2)
        mean_val = np.sum(weights * self.val) / np.sum(weights)
        mean_err = np.sqrt(1 / np.sum(weights))
        self.mean = Var(mean_val, mean_err, short_name=f'{self.short_name}_mean', unit=self.unit)

        return self.mean
    
    def set_unit(self, unit: str):
        '''Set unit for the Var instance.'''
        self.unit = unit
        self.long_name = f'${self.short_name}\\, (\\mathrm{{{self.unit}}})$' if self.unit else f'${self.short_name}$'

    def set_lname(self, short_name: str, unit: str = None):
        '''Set long name for matplotlib axes labels.'''
        self.short_name = short_name
        self.unit = unit
        self.long_name = f'${self.short_name}\\, (\\mathrm{{{self.unit}}})$' if self.unit else f'${self.short_name}$'

    def __str__(self):
        return ufmt(self, apx='P')
    
    def ufmt(self, apx='L') -> str:
        '''Formats all values in the Var instance using scalar_ufmt function, returns a formatted string.
        
        appendix apx: e for exponential notation, S for notation by norm: "value(uncertainty)", 
        L for LaTeX: "value \\pm uncertainty", P for pretty print: "value ± uncertainty".'''

        return ufmt(self, apx=apx)
    
    def latex(self) -> str:
        '''Returns a LaTeX formatted string of the Var instance.'''
        return ufmt(self, apx='L')
    
    def norm(self) -> str:
        '''Returns a string formatted by norm: "value(uncertainty)".'''
        return ufmt(self, apx='S')    

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
            unit = f'{self.unit}*{other.unit}' if self.unit and other.unit else self.unit or other.unit
        else:
            new_unc = self.unc * other
            unit = self.unit

        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), self.short_name, unit)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Var):
            new_unc = self.unc / other.unc
            unit = f'{self.unit}/{other.unit}' if self.unit and other.unit else self.unit or other.unit
        else:
            new_unc = self.unc / other
            unit = self.unit

        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), self.short_name, unit)
    
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
    
    def sin(self):
        new_unc = unp.sin(self.unc)
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), f'\\sin{{{self.short_name}}}', self.unit)
    
    def cos(self):
        new_unc = unp.cos(self.unc)
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), f'\\cos{{{self.short_name}}}', self.unit)
    
    def tan(self):
        new_unc = unp.tan(self.unc)
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), f'\\tan{{{self.short_name}}}', self.unit)
    
    def radians(self):
        new_unc = self.unc * np.pi / 180
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), f'{self.short_name}', 'rad')
    
    def ensure_radians(self):
        if self.unit == 'rad':
            return self
        else:
            return self.radians()

    def degrees(self):
        new_unc = self.unc * 180 / np.pi
        return Var(unp.nominal_values(new_unc), unp.std_devs(new_unc), f'{self.short_name}', '\\degree')
    
    def ensure_degrees(self):
        if self.unit == '\\degree':
            return self
        else:
            return self.degrees()
    
    def __len__(self):
        return len(self.unc)
    
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
############## NonErrorVar class #######################
class NonErrorVar(Var):
    
    def __init__(self, values: int, short_name: str = '', unit:str = '', fmt: str = '.3f'):
        super().__init__(values, short_name=short_name, unit = unit)
        self.val = np.atleast_1d(values)
        self.unc = np.atleast_1d(values)

        self.short_name = short_name
        self.unit = unit
        self.long_name = f'${self.short_name}\\, (\\mathrm{{{self.unit}}})$' if self.unit else f'${self.short_name}$'
        self.fmt = fmt

    def __str__(self):
        return self.ufmt(self.fmt)
    
    def scalar_ufmt(self, x: float):
        return f'{x:{self.fmt}}'
    
    def ufmt(self):
        if len(self.unc) == 1: # scalar value and not pretty print
            string = f'{self.short_name} = {self.scalar_ufmt(self.unc[0], self.fmt)}'
        else: # 1d array of values
            string = f'{self.short_name} = ({", ".join(self.scalar_ufmt(x, self.fmt) for x in self.unc)})'

        if self.unit:
            string += f' {self.unit}'

        return string
############### Function class #########################
class F:
    '''A collection of common fitting functions.'''
    def const(x, constant):
        return constant * np.ones_like(x)
    
    def direct(x, slope):
        return slope * x
    
    def linear(x, intercept, slope):
        return slope * x + intercept
    
    def quadratic(x, a0, a1, a2):
        return a0 + a1 * x + a2 * x**2
    
    def pure_quadratic(x, a2):
        return a2 * x**2
    
    def cubic(x, a0, a1, a2, a3):
        return a0 + a1 * x + a2 * x**2 + a3 * x**3
    
    def quartic(x, a0, a1, a2, a3, a4):
        return a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    
    def polynomial(x, *coeffs):
        '''coeffs are in increasing order, i.e., coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...'''
        #new_coeffs = [c.val[0] for c in coeffs]
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

    def dispersion(x, lambda0, a, n0):
        return n0 + a / (x + lambda0)

############### Relation class #########################
class Rel:
    '''This class provides a relation between two variables (from class Var):
    independent x and dependent y defined by specific function. That function
    is then used for fitting the data: y = func(x).
    
    Atributes: independent Var, dependent Var, function, coefficients (.coeffs: list[Var]), 
    covariant matrix (.cov).
    
    Methods: fit, plot_data, plot_function, show_equation.'''

    def __init__(self, x: Var, y: Var, function: callable = None, color='black', shape='s'):
        self.x = x
        self.y = y
        self.func = function
        self.color = color
        self.shape = shape
    
    def set_degree(self, degree: int = None):
        '''Sets the degree of the polynomial function for fitting.'''
        if degree is not None:
            self.degree = degree
        elif self.func in [F.const, F.direct, F.linear, F.quadratic, F.pure_quadratic, F.cubic, F.quartic]:
            self.degree = {'const': 0, 'direct': 1, 'linear': 1, 'quadratic': 2, 'pure_quadratic': 2,
                           'cubic': 3, 'quartic': 4}[self.func.__name__]
        else:
            self.degree = None
            if self.func is None:
                raise ValueError('Degree or function must be specified for fitting functions.')

    def fit(self, p0: list = None, get_unit: bool = True):
        '''Fits y over x (both Var instances) using the provided function.
        Adds these attributes to Relation: coefficients (.coeffs: list[Var]), covariant matrix (.cov).'''

        if self.func is None:
            raise ValueError('No fitting function defined for the Relation.')
        if self.func in [F.const, F.direct, F.linear, F.quadratic, F.pure_quadratic, F.cubic, 
                         F.quartic, F.polynomial]:
            self.set_degree(len(p0)-1 if p0 else None)
            params, pcov = np.polynomial.Polynomial.fit(self.x.val, self.y.val, deg=self.degree, full=True,
                                                        w=1/self.y.err**2)
            params = params.convert().coef
            variances = pcov[2]
            coeffs = [Var(param, np.sqrt(variances[i]), f'c_{i}', '') for i, param in enumerate(params)]

            self.cov = variances
            self.coeffs = coeffs
        else:
            coeff_values, cov_matrix  = curve_fit(self.func, self.x.val, self.y.val, sigma=self.y.err,
                                              absolute_sigma=True, p0=p0)
            #alpha = 0.05  # 95% confidence interval
            #t_val = t.ppf(1.0-alpha/2, max(0, len(x.val)-len(coeff_values)))
            coeff_errors = np.sqrt(np.diag(cov_matrix)) #* np.abs(t_val)
            fitted_coeffs = [Var(coeff_values[i],coeff_errors[i], 
                             short_name=f'c_{i}') for i in range(len(coeff_values))]

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
    
    def plot_data(self, ax, err: tuple = (1,1), connect=False, smooth:int = 0, label: str ='naměřené hodnoty', 
                  scale: float = 1, zorder: int = None):
        '''Plots self.x and self.y data with optional error bars (err: tuple with (xerr, yerr)).
        
        connect for line-connecting data points, smooth for spline interpotation,
        label for legend, scale for marker size and zorder for plotting order'''
        #self.data_label = label
        match err:
            case (1, 1): # x_error_bar and y_error_bar
                    self.data_curve = ax.errorbar(self.x.val, self.y.val, xerr=self.x.err, yerr=self.y.err, 
                                marker=self.shape, linewidth=1, color=self.color, markersize=5*scale, 
                                capsize=3, label=label, linestyle='', zorder=zorder)
            case (1, 0): # just x_error_bar
                    self.data_curve = ax.errorbar(self.x.val, self.y.val, xerr=self.x.err, marker=self.shape, linewidth=1, 
                                color=self.color, markersize=5*scale, capsize=3, label=label, 
                                linestyle='', zorder=zorder)
            case (0, 1): # just y_error_bar
                    self.data_curve = ax.errorbar(self.x.val, self.y.val, yerr=self.y.err, marker=self.shape, 
                                linewidth=1, color=self.color, markersize=5*scale, capsize=3, 
                                label=label, linestyle='', zorder=zorder)
            case (0, 0): # none of the errorbars
                self.data_curve = ax.scatter(self.x.val, self.y.val, marker=self.shape, s=25*scale, 
                           color=self.color, linewidth=1, label=label, zorder=zorder)

        if connect:
            ax.plot(self.x.val, self.y.val, color='black')

        if smooth:
            # Create a smooth line using spline interpolation
            x_smooth = np.linspace(min(self.x.val), max(self.x.val), 300)
            spline = make_interp_spline(self.x.val, self.y.val, k=smooth)  # "smooth" order spline
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color=self.color, linestyle='--')

        ax.set_xlabel(self.x.long_name)
        ax.set_ylabel(self.y.long_name)


    def plot_fit(self, ax, label: str = 'fitovaná přímka', linestyle: str = ':', linewidth: float = 1.5):
        '''Plots fitted funcion using coefficients from self.fit() method.
        
        label for legend, linestyle and linewidth are parameters for the fit line.'''
        if self.func is None:
            raise ValueError('No fitting function defined for the Relation.')
        if not hasattr(self, 'coeffs'):
            raise ValueError('No fitted coefficients found. Please run the fit() method first.')

        #self.fit_label = label
        x = np.linspace(min(self.x.val), max(self.x.val), 200)

        self.fit_curve = ax.plot(x, self.func(x, *[c.val for c in self.coeffs]), linestyle=linestyle, 
                color=self.color, linewidth=linewidth, label=label)[-1]
        ax.set_xlabel(self.x.long_name)
        ax.set_ylabel(self.y.long_name)


    def show_equation(self, ax, format: str = '.3f', combined = True):
        '''Displays the fitted equation in the legend.
        
        format to specify wished decimal digits of coefficients (e.g. '.3f' for 3 decimal digits)'''
        match self.func:
            case F.const:
                eq_string = f'${self.y.short_name} = {self.coeffs[0].val[0]:{format}}$'
            case F.direct:
                eq_string = f'${self.y.short_name} = {self.coeffs[0].val[0]:{format}} \\cdot {self.x.short_name}$'
            case F.linear:
                eq_string = f'${self.y.short_name} = {self.coeffs[1].val[0]:{format}} \\cdot {self.x.short_name} + \
                {self.coeffs[0].val[0]:{format}}$'
            case F.quadratic:
                eq_string = f'${self.y.short_name} = {self.coeffs[2].val[0]:{format}} \\cdot {self.x.short_name}^2 + \
                {self.coeffs[1].val[0]:{format}} \\cdot {self.x.short_name} + {self.coeffs[0].val[0]:{format}}$'
            case F.dispersion:
                if self.coeffs[0].val[0] < 0:
                    sign = '-'
                    lambda0 = -self.coeffs[0].val[0]
                eq_string = f'${self.y.short_name} = {self.coeffs[2].val[0]:{format}} + \
                    \\frac{{{self.coeffs[1].val[0]:{format}}}} \
                    {{{self.x.short_name} {sign} {lambda0:{format}}}}$'
            case _:
                eq_string = f'Fitted function: {self.func.__name__} with coefficients: ' + \
                ', '.join(f'{c.short_name}={c.val[0]:{format}}' for c in self.coeffs)
        
        # display combined handle but also self.label to datapoints
        combined_handle = Line2D([], [], color=self.color, marker=self.shape, linestyle=':', label=eq_string)
        eq_handle = Line2D([], [], color=self.color, marker='', linestyle='', label=eq_string)
        if combined:
            ax.legend(handles=[combined_handle])
        else:
            self.fit_curve.set_label(f'{self.fit_curve.get_label()}:')
            ax.legend(handles=[self.data_curve, self.fit_curve, eq_handle])
        
############### Measuring tool uncertainty class #######
class MeasureUnc:
    '''Class for storing uncertainty values of certaint measuring tool.
    Variable error is 'percents of measured value', constant error is 'number of digits' (err_type = 'digit')
    or 'percents of range' (err_type = 'range').

    Attributes: err_type (str: 'digit' | 'range'), unit (str), data (pd.DataFrame with columns: 
    ranges, resolution, variable_error and constant_error).'''

    def __init__(self, err_type: str, unit: str, data: pd.DataFrame):
        self.err_type = err_type
        self.unit = unit
        data.columns = ['ranges', 'resolution', 'variable_error', 'constant_error']
        self.data = data
    
    def convert_units(self, to_unit: str):
        '''Converts the units of the measuring tool uncertainty to match 
        the unit of the measured variable (to_unit).'''

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

        if self.err_type == 'digit':
            const_base = self.data.resolution.to_numpy()
        if self.err_type == 'range':
            const_base = self.data.ranges.to_numpy()
            self.data.constant_error = self.data.constant_error / 100 # convert from percent to fraction
        
        for i, value in enumerate(var.val):
            net = value < self.data.ranges
            row_idx = self.data.ranges[net].idxmin() # first range larger than value

            # variable error: (percents of measured value)
            errors[i] += np.abs(value) * (self.data.variable_error[row_idx] / 100.0) 
            # constant error: (digits=)resolution/range * constant error
            errors[i] += const_base[row_idx] * self.data.constant_error[row_idx] 

        return Var(var.val, errors, var.short_name, var.unit)

    def __str__(self):
        nrows, _ = self.data.shape
        if self.err_type == 'digit':
            ncols = 3
            const_apx = 'dgt'
        else: # 'err_type == 'range'
            ncols = 2
            const_apx = '\\%'

        string = f'Measuring tool uncertainty (err_type: {self.err_type}, unit: {self.unit}):\n'
        string += '\\begin{table}[!htbp]' + '\n' + \
        '\\centering' + '\n' + \
        '\\caption{CAPTION}' + '\n' + \
        '\\begin{tabular}{' + 'c'*ncols + '}' + '\n' + \
        ' \\toprule' + '\n' + \
        ' rozsah & nejistota \\\\' + '\n' + \
        ' \\midrule' + '\n'
        for i in range(nrows):
            # add ranges
            string += f' ${self.data["ranges"][i]} \\mathrm{{{self.unit}}}$ &'
            # add resolution value if 'digit' error type
            if self.err_type == 'digit':
                string += f' ${self.data["resolution"][i]} \\mathrm{{{self.unit}}}$ &'
            # add variable and constant error
            string += f' ${self.data["variable_error"][i]} \\%$ &'
            string += f' ${self.data["constant_error"][i]} {const_apx}$ \\\\' + '\n'

        string += ' \\bottomrule' + '\n'
        string += '\\end{tabular}' + '\n'
        string += '\\label{tab:LABEL}' + '\n'
        string += '\\end{table}' + '\n'

        string += '\n'

        return string

########################################################
########### other useful functions #####################

def read_excel(file_path, sheet_name='Sheet1', cells='A1:Z100', header = 0):
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
def excel_to_latex(file_path, sheet_name='Sheet1', cells='A1:Z100', header = 0, format='.2f'):
    '''Reads an Excel file and returns a LaTeX formatted table as a string.'''
    df = read_excel(file_path, sheet_name=sheet_name, cells=cells, header=header)
    cols = df.columns.to_list()
    nrows, ncols = df.shape

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
def excel_to_latex_2(file_path, sheet_name='Sheet1', cells='A1:Z100', header = 0, format='.3f', 
                     show_errors = True, apx='N') -> str:
    '''Reads an Excel file and returns a LaTeX formatted table as a string.
    Designed for alternating value and error columns.
    
    If show_errors, print values with errors in format of apx:
    e for exponential notation, S for notation by norm: "value(uncertainty)", 
    L for LaTeX: "value \\pm uncertainty", P for pretty print: "value ± uncertainty"
    
    If not show_errors, print only values. Adds recommendation for unit change,
    for avoiding exponential notation.'''
    
    df = read_excel(file_path, sheet_name=sheet_name, cells=cells, header=header)
    nrows, ncols = df.shape
    #if ncols % 2 != 0:
    #    raise ValueError('Number of columns must be even, with alternating value and error columns.')
    col_names = df.columns.to_list()
    err_indices = []
    for i in range(len(col_names)):
        if col_names[i].startswith('sigma'):
            err_indices.append(i)   # index of error column
    
    val_col_names = [col_names[i] for i in range(ncols) if i not in err_indices] # columns with values

    string = f'\n' + \
    '\\begin{table}[!htbp]' + '\n' + \
    '\\centering' + '\n' + \
    '\\caption{CAPTION}' + '\n' + \
    '\\begin{tabular}{' + 'c'*(ncols-len(err_indices)) + '}' + '\n' + \
    ' \\toprule' + '\n' + \
    ' ' +' & '.join(val_col_names) + ' \\\\' + '\n' + \
    ' \\midrule' + '\n'
    if show_errors:
       for i in range(nrows):
            for j in range(ncols):
                if j+1 in err_indices: # skip, next error column will be processed together
                    continue
                if j in err_indices: # error column
                    string += f' ${scalar_ufmt(ufloat(df.iloc[i,j-1], df.iloc[i,j]), apx)}$ '
                else: # value column
                    if format is not None: # normal value column without error
                        string += f' ${df.iloc[i,j]:{format}}$ '
                    else: # leave value as is
                        string += f' ${df.iloc[i,j]}$ '

                if j == ncols - 1: # last column, add newline
                    string += ' \\\\' + '\n'
                else: # add column separator
                    string += '&'
    else:
        for i in range(nrows):
            if format is not None:
                string += ' $' + '$ & $'.join(f'{df.iloc[i,j]:{format}}' for j in range(ncols) if j not in err_indices) + '$ \\\\' + '\n'
            else:
                string += ' $' + '$ & $'.join(f'{df.iloc[i,j]}' for j in range(ncols) if j not in err_indices) + '$ \\\\' + '\n'
    string += ' \\bottomrule' + '\n'
    string += '\\end{tabular}' + '\n'
    string += '\\label{tab:LABEL}' + '\n'
    string += '\\end{table}' + '\n'

    def find_order(x: list[float]):
        max_val = max(x)
        if max_val == 0:
            return 0
        mag = int(np.floor(np.log10(max_val)))
        k = max_val / (10 ** mag)
        if k <= 1.9:
            highest_mag = 1
        else:
            highest_mag = 0
        return mag - highest_mag

    for i in err_indices: # for each error column:
        change_by_order = find_order(df.iloc[:, i].to_list())
        
        if change_by_order > 0:
            string += f'% RECOMMENDATION: Consider changing unit of "{col_names[i-1]}"' + \
                      f' by {change_by_order} orders, at least {int(np.ceil(change_by_order / 3))} 3-orders.\n'

    return string

########################################################
def scalar_ufmt(x: ufloat, apx='N'):
    '''Work only for scalar values!
    
    Rounds value and error of an ufloat number to appropriate significant figures.
    Returns formatted string. If first significant figure of error is 1 (k <= 1.9), uses two significant digits.
    Appendix (apx) sets formatting options: e for scientific exponential notation,
    S for notation by norm: "value(uncertainty)", L for LaTeX: "value \\pm uncertainty"
    N for notation by norm with LaTeX formatting.'''

    if apx == 'N':
        apx = 'S'
        convert_exponent = True
    else:
        convert_exponent = False

    if x.std_dev == 0:
        return f'{x:.2u{apx}}'
    
    # assume error = k * 10^mag
    mag = int(np.floor(np.log10(x.std_dev)))
    k = x.std_dev / (10 ** mag)
    sig_fig = 2 if k <= 1.9 else 1

    string = f'{x:.{sig_fig}u{apx}}'

    # convert exponent to LaTeX notation if scientific notation is used
    if convert_exponent and 'e' in string:
        value_str, exp_str = string.split('e')
        exp = int(exp_str)
        string = f'{value_str} \\times 10^{{{exp}}}'

    return string

#########################################################
def ufmt(var: Var, apx='N') -> str:
    '''Formats all values in the Var instance using scalar_ufmt function, returns a formatted string.
    
    appendix apx: e for exponential notation, S for notation by norm: "value(uncertainty)", 
    L for LaTeX: "value \\pm uncertainty", P for pretty print: "value ± uncertainty".'''

    if isinstance(var, NonErrorVar):
        return var.ufmt()
    
    if len(var.unc) == 1 and not apx=='P': # scalar value and not pretty print
        string = f'{var.short_name} = {scalar_ufmt(var.unc[0], apx)}'
    else: # 1d array of values or pretty print
        string = f'{var.short_name} = ({", ".join(scalar_ufmt(x, apx) for x in var.unc)})'

    if var.unit:
        string += f' {var.unit}'

    return string

#########################################################
def nice_print(var: Var):
    '''Prints ufloat number with nicely formatted value and error.'''
    print(ufmt(var, apx='P'))

#########################################################
def latex_print(var: Var):
    '''Prints ufloat number with LaTeX formatted value and error.'''
    print(ufmt(var, apx='L'))

#########################################################
def norm_print(var: Var):
    '''Prints ufloat number with value and error formatted by norm: "value(uncertainty)".'''
    print(ufmt(var, apx='S'))

#########################################################
def best_print(var: Var):
    '''Prints Var instance with format by norm: "value(uncertainty)". If exponential notation,
    converts to LaTeX format.'''
    print(ufmt(var, apx='N'))

#########################################################
def to_table(*args, apx='N'):
    '''Converts Var instances to a formatted LaTeX table. Uses ufmt for formatting, 
    defaultly writes uncertainties in format specified by norm (N) with LaTeX exponential notation.'''
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

#########################################################
def to_table_2(*args, apx='N') -> str:
    '''Converts Var instances to a formatted LaTeX table. Uses ufmt for formatting, 
    defaultly writes uncertainties in format specified by norm (N) with LaTeX exponential notation.'''
    # df = pd.DataFrame()
    # for var in args:
    #     df[var.long_name] = [scalar_ufmt(x, apx=apx) for x in var.unc]
    # return df.to_latex(index=False, column_format=len(args)*'c')

    string = '\n'
    string += '\\begin{table}[!htbp]\n'
    string += '\\centering\n'
    string += '\\caption{CAPTION}\n'
    string += '\\begin{tabular}{' + 'c'*len(args) + '}\n'
    string += ' \\toprule\n'
    string += ' '
    for i in range(len(args)):
        if isinstance(args[i], Var) or isinstance(args[i], NonErrorVar):
            string += str(args[i].long_name)
        elif isinstance(args[i], pd.Series):
            string += str(args[i].name)
        else:
            string += f'NAME_{i}'

        if i == len(args) - 1: # last column, add newline
            string += ' \\\\\n'
        else: # add column separator
            string += ' & '

    string += ' \\midrule\n'
    for i in range(len(args[0])):
        string += ' '
        for j in range(len(args)):
            if isinstance(args[j], NonErrorVar):
                string += f'${args[j].scalar_ufmt(args[j].unc[i])}$'
            elif isinstance(args[j], Var):
                string += f'${scalar_ufmt(args[j].unc[i], apx=apx)}$'
            
            elif isinstance(args[j], pd.Series):
                if isinstance(args[j].iloc[i], float) or isinstance(args[j].iloc[i], int):
                    string += f'${args[j].iloc[i]}$'
                else:
                    string += str(args[j].iloc[i])
            else:
                string += str(args[j][i])
            
            if j == len(args) - 1: # last column, add newline
                string += ' \\\\\n'
            else: # add column separator
                string += ' & '

    string += ' \\bottomrule\n'
    string += '\\end{tabular}\n'
    string += '\\label{tab:LABEL}\n'
    string += '\\end{table}\n'

    return string

#########################################################
def sin(var: Var) -> Var:
    return var.sin()
#########################################################
def cos(var: Var) -> Var:
    return var.cos()
#########################################################
def tan(var: Var) -> Var:
    return var.tan()
#########################################################
def exp(var: Var) -> Var:
    return var.exp()
#########################################################
def log(var: Var) -> Var:
    return var.log()

#########################################################
#################### LEGACY CODE ########################

def fit_curve(x: Var, y: Var, func=F.linear, p0: list =None):
    '''!!! NOT SUPPORTED ANYMORE!!! -> use Rel class and its fit() method instead.

    Returns the fitted coefficients as uarray.'''
    #alpha = 0.05  # 95% confidence interval
    coeff_values, cov_matrix  = curve_fit(func, x.val, y.val, sigma=y.err, absolute_sigma=True, p0=p0)
    #t_val = t.ppf(1.0-alpha/2, max(0, len(x.val)-len(coeff_values)))
    coeff_errors = np.sqrt(np.diag(cov_matrix)) #* np.abs(t_val)
    fitted_coeffs = Var([*coeff_values],[*coeff_errors], short_name='fit_coeffs')

    return fitted_coeffs

#########################################################
def plot(fig, ax, x, y, fit_coeffs=None, func=F.linear, xerr=True, yerr=True, fit_curve=True, 
         data_label='naměřené hodnoty', fit_label='teoretická závislost', equation=False):

    '''!!!NOT SUPPORTED ANYMORE!!! -> use Rel class and its plot_data() and plot_fit() methods instead.
    
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

#########################################################
def plot_data(ax, x: Var, y: Var, err: tuple = (1,1), label='naměřené hodnoty', scale=1, marker='s', color='black'):
    '''!!!NOT SUPPORTED ANYMORE!!! -> use Rel class and its plot_data() method instead.'''
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

#########################################################
def plot_fit(ax, x: Var, coeffs: Var, func=F.linear, label='fitovaná přímka', linestyle=':', color='black'):
    '''!!!NOT SUPPORTED ANYMORE!!! -> use Rel class and its plot_fit() method instead.'''
    ax.plot(x.val, func(x.val, *coeffs.val), linestyle=linestyle, color=color, linewidth=1.5, label=label)

#########################################################
################## testing ##############################

if __name__ == "__main__":

#########################################################
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
#########################################################
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
#########################################################
    # scalar_ufmt
    '''
    print(scalar_ufmt(ufloat(4.965,0.08)))
    print(scalar_ufmt(ufloat(5.334,0.00134),apx=''))
    print(scalar_ufmt(ufloat(4.222,0.00190)))
    print(scalar_ufmt(ufloat(149.5,56.6)))
    print(scalar_ufmt(ufloat(0.001495,0.000566), 'eL'))
    print(scalar_ufmt(ufloat(14.95,0.566), 'L'))
    '''
#########################################################
    # polynomial fit
    '''
    cff = fit_curve(x, y, func=F.polynomial, p0=[1,2,3])
    print('Fitted coeffs:', cff)
    '''
#########################################################
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
#########################################################
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

#########################################################
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
#########################################################
    # excel_to_latex
    '''
    latex_table = excel_to_latex('2zs/uloha8/uloha8.xlsx', sheet_name='List1', cells='A2:C7', format='.3f')
    print(latex_table)
    '''
#########################################################
    # excel_to_latex_2
    '''
    print('old version:')
    print(excel_to_latex('excel_to_latex_test.ods', 'Sheet1', 'A1:H25'))
    print('without errors:')
    print(excel_to_latex_2('excel_to_latex_test.ods', 'Sheet1', 'A1:H25', show_errors=False))
    print('with errors:')
    print(excel_to_latex_2('excel_to_latex_test.ods', 'Sheet1', 'A1:H25', show_errors=True))
    #'''
#########################################################
    # formats
    '''
    a = ufloat(1.2345, 0.06789)
    b = ufloat(0.0012345, 0.00006789)
    c = ufloat(12345, 678.9)
    print('scalar_ufmt:')
    print(scalar_ufmt(a, 'S'))
    print(scalar_ufmt(b, 'S'))
    print(scalar_ufmt(c, 'S'))

    print(scalar_ufmt(a, 'N'))
    print(scalar_ufmt(b, 'N'))
    print(scalar_ufmt(c, 'N'))

    print(scalar_ufmt(a, 'L'))
    print(scalar_ufmt(b, 'L'))
    print(scalar_ufmt(c, 'L'))

    print(scalar_ufmt(a, 'P'))
    print(scalar_ufmt(b, 'P'))
    print(scalar_ufmt(c, 'P'))

    A = Var(a.nominal_value, a.std_dev, short_name='A', unit='m')
    B = Var(b.nominal_value, b.std_dev, short_name='B', unit='s')
    C = Var(c.nominal_value, c.std_dev, short_name='C', unit='J')
    print('ufmt:')
    print(ufmt(C, 'S'))
    print(ufmt(C, 'L'))
    print(ufmt(C, 'P'))
    print(A, B, C, sep=', ')
    best_print(A)
    best_print(B)
    best_print(C)
    #'''

#####################################################
    # polynomial
    '''
    x=5
    print(F.polynomial(x,1,2,3))
    #'''
#########################################################
    # None errors
    #'''
    a = NonErrorVar(1.44, 'var_a', 'mm')
    print(a)
    #'''


    print('All done!')
