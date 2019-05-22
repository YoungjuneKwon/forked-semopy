'''This module implements polychoric and polyserial correlations. It is assumed
for internal usage only.'''
from scipy.optimize import minimize, minimize_scalar
from itertools import chain, product, combinations
from scipy.stats import norm, mvn
from .utils import cor
import numpy as np


def bivariate_cdf(lower, upper, corr, means=[0, 0], var=[1, 1]):
    """Estimates an integral of bivariate pdf given integration lower and 
    upper limits. Consider using relatively big (i.e. 20 if using default mean
    and variance) lower and/or upper bounds when integrating to/from infinity.
    
    Keyword arguments:
        
        lower -- Lower integration bounds.
        
        upper -- Upper integration bounds.
        
        corr  -- Correlation coefficient between variables.
        
        means -- Mean values of variables (assumed to be [0, 0] by default).
        
        var   -- Variances of variables (assumed to be [1, 1] by default).
        
    Returns:
        
        P(lower[0] < x < upper[0], lower[1] < y < upper[1]).
    """
    s = np.array([[var[0], corr], [corr, var[1]]])
    return mvn.mvnun(lower, upper, means, s)[0]

def univariate_cdf(lower, upper, mean=0, var=1):
    """Estimates an integral of univariate pdf given integration lower and 
    upper limits. Consider using relatively big (i.e. 20 if using default mean
    and variance) lower and/or upper bounds when integrating to/from infinity.
    
    Keyword arguments:
        
        lower -- Lower integration bound.
        
        upper -- Upper integration bound.
        
        mean -- Mean value of variable (assumed to be 0 by default).
        
        var  -- Variance of variable (assumed to be 1 by default).
        
    Returns:
        
        P(lower < x < upper).
    """
    return mvn.mvnun([lower], [upper], [mean], [var])[0]

def estimate_intervals(x, inf=10):
    """Estimates intervals of the polytomized underlying latent variable.
    
    Keyword arguments:
        
        x   -- An array of values the ordinal variable.
        
        inf -- A numerical infinity substitute (10 by default).
        
    Returns:
        
        An array containing polytomy intervals and an array containing indices
        of intervals correspoding to each entry in x.
    """
    x_f = x[~np.isnan(x)]
    u, counts = np.unique(x_f, return_counts=True)
    sz = len(x_f)
    cumcounts = np.cumsum(counts[:-1])
    u = [np.where(u == sample)[0][0] + 1 for sample in x]
    return list(chain([-inf], (norm.ppf(n / sz) for n in cumcounts), [inf])), u

def polyserial_corr(x, y, x_mean=None, x_var=None, x_z=None, x_pdfs=None,
                    y_ints=None, scalar=True):
    """Estimates polyserial correlation between continious variable x and
    ordinal variable y.
    
    Keyword arguments:
        
        x -- Data sample corresponding to x.
        
        y -- Data sample corresponding to y.
        
        x_mean -- A mean value of x (calculated if not provided).
        
        x_var  -- A variance of x (calculated if not provided).
        
        x_z    -- A standartized x (calculated if not provided).
        
        x_pdfs -- x's logpdf sampled at each point.
        
        y_ints -- Polytomic intervals of an underlying latent variable
                  correspoding to y (calculated if not provided) as returned
                  by estimate_intervals.
                  
        scalar -- If true minimize_scalar is used instead of SLSQP.
        .
    Returns:
        
        A polyserial correlation coefficient for x and y.
    """
    if x_mean is None:
        x_mean = np.nanmean(x)
    if x_var is None:
        x_var = np.nanvar(x)
    if y_ints is None:
        y_ints = estimate_intervals(y)
    if x_z is None:
        x_z = (x - x_mean) / x_var
    if x_pdfs is None:
        x_pdfs = norm.logpdf(x, x_mean, x_var)
    ints, inds = y_ints
    def transform_tau(tau, rho, z):
        return (tau - rho * z) / np.sqrt(1 - rho ** 2)
    def sub_pr(k, rho, z):
        i = transform_tau(ints[k], rho, z)
        j = transform_tau(ints[k - 1], rho, z)
        return univariate_cdf(j, i)
    def calc_likelihood(rho):
        return -sum(pdf + np.log(sub_pr(ind, rho, z))
                    for z, ind, pdf in zip(x_z, inds, x_pdfs))
    def calc_likelihood_derivative(rho):
        def sub(k, z):
            i = transform_tau(ints[k], rho, z)
            j = transform_tau(ints[k - 1], rho, z)
            a = norm.pdf(i) * (ints[k] * rho - z)
            b = norm.pdf(j) * (ints[k - 1] * rho - z)
            return a - b
        t = (1 - rho ** 2) ** 1.5
        return -sum(sub(ind, z) / sub_pr(ind, rho, z)
                   for x, z, ind in zip(x, x_z, inds) if not np.isnan(x)) / t
    if not scalar:
        res = minimize(calc_likelihood, [0.0], jac=calc_likelihood_derivative,
                       method='SLSQP', bounds=[(-1.0, 1.0)]).x[0]
    else:
        res = minimize_scalar(calc_likelihood, bounds=(-1, 1),
                              method='bounded').x
    return res

def polychoric_corr(x, y, x_ints=None, y_ints=None):
    """Estimates polyserial correlation between ordinal variables x and y.
    
    Keyword arguments:
        
        x      -- Data sample corresponding to x.
        
        y      -- Data sample corresponding to y.
        
        x_ints -- Polytomic intervals of an underlying latent variable
                  correspoding to y (calculated if not provided) as returned
                  by estimate_intervals.
                  
        y_ints -- Polytomic intervals of an underlying latent variable
                  correspoding to y (calculated if not provided) as returned
                  by estimate_intervals.
                  
    Returns:
        
        A polychoric correlation coefficient for x and y.
    """
    if x_ints is None:
        x_ints = estimate_intervals(x)
    if y_ints is None:
        y_ints = estimate_intervals(y)
    x_ints, x_inds = x_ints
    y_ints, y_inds = y_ints
    p, m = len(x_ints) - 1, len(y_ints) - 1
    n = np.zeros((p, m))
    for a, b in zip(x_inds, y_inds):
        if not (np.isnan(a) or np.isnan(b)):
            n[a - 1, b - 1] += 1
    def calc_likelihood(r):
        return -sum(np.log(bivariate_cdf([x_ints[i], y_ints[j]],
                                 [x_ints[i + 1], y_ints[j + 1]], r)) * n[i, j]
                    for i in range(p) for j in range(m))
    return minimize_scalar(calc_likelihood, bounds=(-1, 1), method='bounded').x
                

def hetcor(data, ords=None):
    """Computes a heterogenous correlation matrix.
    
    Keyword arguments:
        
        data -- Either a pandas DataFrame or a numpy matrix of sample data.
        
        ords -- Names of ordinal variables if data is DataFrame or indices of
                ordinal numbers if data is np.array. If ords are None (default)
                then ordinary variables will be determined automatically.
                
    Returns:
        
        A heterogenous correlation matrix.
    """  
    if type(data) is np.array:
        cov = cor(data)
        if ords is None:
            ords = set()
            for i in range(data.shape[1]):
                if len(np.unique(data[:, i])) / data.shape[0] < 0.3:
                    ords.add(i)
        conts = set(range(data.shape[1])) - set(ords)
    else:
        cov = data.corr()
        if ords is None:
            ords = set()
            for var in data:
                if len(data[var].unique()) / len(data[var]) < 0.3:
                    ords.add(var)
        conts = set(data.columns) - set(ords)
    c_means = {v: np.nanmean(data[v]) for v in conts}
    c_vars = {v: np.nanvar(data[v]) for v in conts}
    c_z = {v: (data[v] - c_means[v]) / c_vars[v] for v in conts}
    c_pdfs = {v: norm.logpdf(data[v], c_means[v], c_vars[v]) for v in conts}
    o_ints = {v: estimate_intervals(data[v]) for v in ords}

    for c, o in product(conts, ords):
        cov[c][o] = polyserial_corr(data[c], data[o], x_mean=c_means[c],
                                    x_var=c_vars[c], x_z=c_z[c],
                                    x_pdfs=c_pdfs[c], y_ints=o_ints[o])
        cov[o][c] = cov[c][o]
    for a, b in combinations(ords, 2):
        cov[a][b] = polychoric_corr(data[a], data[b], o_ints[a], o_ints[b])
        cov[b][a] = cov[a][b]
    return cov
