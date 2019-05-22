'''Module contains all methods related to calculating parameters' estimates' 
statistics and models' fit indices.'''
from .optimizer import Optimizer
from .model import Model
from collections import namedtuple
from scipy.special import multigammaln
from scipy.stats import norm, chi2
from .utils import chol_inv
import pandas as pd
import numpy as np

ParameterStatistics = namedtuple('ParametersStatistics',
                                 ['value', 'se', 'zscore', 'pvalue'])
SEMStatistics = namedtuple('SEMStatistics', ['dof', 'ml', 'fit_val', 
                                             'chi2', 'dof_baseline',
                                             'chi2_baseline', 'rmsea', 'cfi',
                                             'gfi', 'agfi', 'nfi', 'tli',
                                             'aic', 'bic', 'params'])

def get_baseline_model(model, data=None):
    """Retrieves a the baseline model from given model. Baseline model
    here is an independence model where all variables are considered to be 
    independent with zero covariance. Only variances are estimated.
    
    Keyword arguments:
        
        model -- Either string description or Model instance.
        
    Returns:
        
        A baseline model.
    """
    if type(model) is str:
        mod = Model(model, baseline=True)
        if data:
            mod.load_dataset(data)
        return mod
    desc = model.model_description
    mod = Model(desc, baseline=True)
    try:
        if not data:
            data = pd.DataFrame(data=model.raw_data,
                                columns=model.vars['IndsObs'])
        mod.load_dataset(data)
    except AttributeError:
        pass
    return mod

def __get_chi2_base(opt: Optimizer):
    mod = get_baseline_model(opt.model)
    opt_base = Optimizer(mod)
    opt_base.optimize(*opt.last_run[0])
    chi2_base = calculate_chi_square(opt_base)[0]
    return chi2_base, calculate_dof(opt_base)

        
def calculate_gfi(opt: Optimizer, chi2=None, chi2_base=None):
    """Calculates GFI (goodness-of-fit index).
    
    Keyword arguments:
        
        opt       -- Optimizer containing proper parameters' values (not used
                     if chi2 and chi2_base provided).
        
        chi2      -- chi2 statistics for target model.
        
        chi2_base -- chi2 statistics for baseline model.
        
    Returns:
        
        GFI.
    """
    if chi2 is None:
        chi2 = calculate_chi_square(opt)[0]
    if chi2_base is None:
        chi2_base = __get_chi2_base(opt)[0]
    return 1 - chi2 / chi2_base

def calc_gfi(opt: Optimizer, chi2=None, chi2_base=None):
    """Calculates GFI (goodness-of-fit index).
    
    Keyword arguments:
        
        opt       -- Optimizer containing proper parameters' values (not used
                     if chi2 and chi2_base provided).
        
        chi2      -- chi2 statistics for target model.
        
        chi2_base -- chi2 statistics for baseline model.
        
    Returns:
        
        GFI.
    """
    return calculate_gfi(opt, chi2, chi2_base)

def calculate_agfi(opt: Optimizer, dof=None, gfi=None):
    """Calculates AGFI (adjusted goodness-of-fit index).
    
    Keyword arguments:
        
        opt -- Optimizer containing proper parameters' values (not used
               if dof and gfi provided).
        
        dof -- Degrees of freedom.
        
        gfi -- GFI.
        
    Returns:
        
        AGFI.
    """
    if dof is None:
        dof = calculate_dof(opt)
    if gfi is None:
        gfi = calculate_gfi(opt)[0]
    k = len(opt.params)
    return 1 - k * (k + 1) / (2 * dof) * (1 - gfi)

def calc_agfi(opt: Optimizer, dof=None, gfi=None):
    """Calculates AGFI (adjusted goodness-of-fit index).
    
    Keyword arguments:
        
        opt -- Optimizer containing proper parameters' values (not used
               if dof and gfi provided).
        
        dof -- Degrees of freedom.
        
        gfi -- GFI.
        
    Returns:
        
        AGFI.
    """
    return calculate_agfi(opt, dof, gfi)

def calculate_dof(opt: Optimizer):
    """Calculates degrees of freedom.
    
    Keyword arguments:
        
        opt -- Optimizer containing proper parameters' values.
        
    Returns:
        
        Degrees of freedom.
    """
    k = len(opt.model.vars['IndsObs'])
    return k * (k + 1) / 2 - len(opt.params)

def calc_dof(opt: Optimizer):
    """Calculates degrees of freedom.
    
    Keyword arguments:
        
        opt -- Optimizer containing proper parameters' values.
        
    Returns:
        
        Degrees of freedom.
    """
    return calculate_dof(opt)

def calculate_nfi(opt: Optimizer, dof=None, chi2=None, chi2_base=None):
    """Calculates NFI (Normed Fit Index).
    
    Keyword arguments:
    
        opt       -- Optimizer containing proper parameters' values (not used
                     if chi2 and chi2_base provided).
        
        chi2      -- chi2 statistics for target model.
        
        chi2_base -- chi2 statistics for baseline model.
        
    Returns:
        
        NFI.
    """
    if chi2 is None:
        chi2 = calculate_chi_square(opt)[0]
    if chi2_base is None:
        chi2_base = __get_chi2_base(opt)[0]
    return (chi2_base -chi2) / chi2_base

def calc_nfi(opt: Optimizer, dof=None, chi2=None, chi2_base=None):
    """Calculates NFI (Normed Fit Index).
    
        opt       -- Optimizer containing proper parameters' values (not used
                     if chi2 and chi2_base provided).
        
        chi2      -- chi2 statistics for target model.
        
        chi2_base -- chi2 statistics for baseline model.
        
    Returns:
        
        NFI.
    """
    return calculate_nfi(opt, dof, chi2, chi2_base)

def calculate_tli(opt: Optimizer, dof=None, chi2=None, dof_base=None,
                  chi2_base=None):
    """Calculates TLI (Tucker and Lewis Index).
    
    Keyword arguments:
    
        opt       -- Optimizer containing proper parameters' values (not used
                     if chi2, chi2 dof, dof_base_base provided).
        
        dof       -- Degrees of freedom for target mode.
        
        chi2      -- chi2 statistics for target model.
        
        dof_base  -- Degrees of freedom for baseline model.
        
        chi2_base -- chi2 statistics for baseline model.
        
    Returns:
        
        TLI.
    """
    if chi2 is None:
        chi2 = calculate_chi_square(opt)[0]
    if chi2_base is None or dof_base is None:
        chi2_base, dof_base = __get_chi2_base(opt)[0]
    a, b = chi2 / dof, chi2_base / dof_base
    return (b - a) / (b - 1)

def calc_tli(opt: Optimizer, dof=None, chi2=None, dof_base=None,
                  chi2_base=None):
    """Calculates TLI (Tucker and Lewis Index).
    
        opt       -- Optimizer containing proper parameters' values (not used
                     if chi2, chi2 dof, dof_base_base provided).
        
        dof       -- Degrees of freedom for target mode.
        
        chi2      -- chi2 statistics for target model.
        
        dof_base  -- Degrees of freedom for baseline model.
        
        chi2_base -- chi2 statistics for baseline model.
        
    Returns:
        
        TLI.
    """
    return calculate_tli(opt, dof, chi2, dof_base, chi2_base)

def calculate_cfi(opt: Optimizer, dof=None, chi2=None, dof_base=None,
                  chi2_base=None):
    """Calculates CFI (Comparative Fit Index).
    
    Keyword arguments:
        
        opt       -- Optimizer containing proper parameters' values (not used
                     if chi2, chi2 dof, dof_base_base provided).
        
        dof       -- Degrees of freedom for target mode.
        
        chi2      -- chi2 statistics for target model.
        
        dof_base  -- Degrees of freedom for baseline model.
        
        chi2_base -- chi2 statistics for baseline model.
    
    Returns:
    
        TLI.
    """
    if chi2 is None:
        chi2 = calculate_chi_square(opt)[0]
    if dof is None:
        dof = calculate_dof(opt)
    if chi2_base is None or dof_base is None:
        chi2_base, dof_base = __get_chi2_base(opt)
    a = chi2 - dof
    b = chi2_base - dof_base
    return 1 - a / b

def calc_cfi(opt: Optimizer, dof=None, chi2=None, dof_base=None,
                  chi2_base=None):
    """Calculates CFI (Comparative Fit Index).
    
    Keyword arguments:
        
        opt       -- Optimizer containing proper parameters' values (not used
                     if chi2, chi2 dof, dof_base_base provided).
        
        dof       -- Degrees of freedom for target mode
        
        chi2      -- chi2 statistics for target model.
        
        dof_base  -- Degrees of freedom for baseline model.
        
        chi2_base -- chi2 statistics for baseline model.
    
    Returns:
    
        TLI.
    """
    return calculate_cfi(opt, dof, chi2, dof_base, chi2_base)

def calculate_chi_square(opt: Optimizer, dof=None):
    """Calculates chi-square statistics.
    
    Keyword arguments:
    
        opt -- Optimizer containing estimated parameters.
        
        dof -- Degrees of freedom.
    
    Returns:
    
        Chi-Square statistics, p-value.
    """
    if dof is None:
        dof = calculate_dof(opt)
    stat = opt.profiles.shape[0] * opt.last_run[1]
    return stat, 1 - chi2.cdf(stat, dof)

def calc_chi2(opt: Optimizer, dof=None):
    """Calculates chi-square statistics.
    
    Keyword arguments:
    
        opt -- Optimizer containing estimated parameters.
        
        dof -- Degrees of freedom.
    
    Returns:
    
        Chi-Square statistics, p-value.
    """
    return calculate_chi_square(opt, dof)

def calculate_rmsea(opt: Optimizer, chisqr=None, dof=None):
    """Calculates RMSEA statistics;
    
    Keyword arguments:
    
        opt    -- Optimizer containing estimated parameters.
        
        chisqr -- chi2 statistics.
        
        dof    -- Degrees of freedom.
    
    Returns:
    
        RMSEA.
    """
    if chisqr is None:
        chisqr = calculate_chi_square(opt)[0]
    if dof is None:
        dof = calculate_dof(opt)
    if chisqr < dof:
        return 0
    return np.sqrt((chisqr / dof - 1) / (opt.profiles.shape[0] - 1))

def calc_rmsea(opt: Optimizer, chisqr=None, dof=None):
    """Calculates RMSEA statistics;
    
    Keyword arguments:
    
        opt    -- Optimizer containing estimated parameters.
        
        chisqr -- chi2 statistics.
        
        dof    -- Degrees of freedom.
    
    Returns:
    
        RMSEA.
    """
    return calculate_rmsea(opt, chisqr, dof)
    

def calculate_likelihood(opt: Optimizer, dist='wishart'):
    """Calculates Wishart likelihood.
    
    Keyword arguments:
    
        opt  -- Optimizer containing proper parameters' values.
        
        dist -- ML distribution ('normal' or 'wishart')
   
    Returns:
    
        Wishart likelihood.
    """
    sigma, _ = opt.get_sigma()
    if dist == 'wishart':
        p = sigma.shape[0]
        n = opt.model.num_params
        tr = -np.einsum('ij,ji->', opt.mx_cov, np.linalg.pinv(sigma)) / 2
        det_s = (n - p - 1) / 2 * opt.cov_logdet
        det_sigma = - n * np.linalg.slogdet(sigma)[1] / 2
        det = det_s + det_sigma
        c = - n / 2 * np.log(2)
        lngamma = -multigammaln(n / 2, p)
        return tr + det + c + lngamma
    elif dist == 'normal':
        x = opt.profiles
        n = x.shape[0]
        slogdet = n * np.linalg.slogdet(sigma)[1]
        t = sum(x[i][np.newaxis] @ x[i][np.newaxis].T for i in range(n))
        return - (slogdet + np.einsum('ij,ji->', np.linalg.pinv(sigma), t)) / 2
    else:
        raise Exception("Unsupported distribution {}.".format(dist))

def calc_likelihood(opt: Optimizer, dist='wishart'):
    """Calculates Wishart likelihood.
    
    Keyword arguments:
    
        opt  -- Optimizer containing proper parameters' values.
        
        dist -- ML distribution ('normal' or 'wishart')
    
    Returns:
    
        Wishart likelihood.
    """
    return calculate_likelihood(opt, dist)

def calculate_aic(opt: Optimizer, lh=None):
    """Calculates AIC.
    
    Keyword arguments:
    
        opt -- Optimizer containing proper parameters' values.
        
        lh  -- Likelihood in case it was already calculated.
    
    Returns:
    
        AIC.
    """
    if lh is None:
        lh = calculate_likelihood(opt)
    return 2 * (len(opt.params) - lh)

def calc_aic(opt: Optimizer, lh=None):
    """Calculates AIC.
    
    Keyword arguments:
    
        opt -- Optimizer containing proper parameters' values.
        
        lh  -- Likelihood in case it was already calculated.
    
    Returns:
    
        AIC.
    """
    return calculate_aic(opt, lh)

def calculate_bic(opt: Optimizer, lh=None):
    """Calculates BIC.
    
    Keyword arguments:
    
        opt -- Optimizer containing proper parameters' values.
        
        lh  -- Likelihood in case it was already calculated.
    
    Returns:
    
        BIC.
    """
    if lh is None:
        lh = opt.calculate_likelihood(opt.params)
    k, n = len(opt.params), opt.profiles.shape[0]
    return np.log(n) * k - 2 * lh

def calc_bic(opt: Optimizer, lh=None):
    """Calculates BIC.
    
    Keyword arguments:
    
        opt -- Optimizer containing proper parameters' values.
        
        lh  -- Likelihood in case it was already calculated.
    
    Returns:
    
        BIC.
    """
    return calculate_bic(opt, lh)

def calculate_standard_errors(opt: Optimizer, information='expected'):
    """Calculates standard errors.
    
    Keyword arguments:
    
        opt         -- Optimizer containing proper parameters' values.
        
        information -- Whether to use "expected" or "observed" Fisher information
                       matrix.
    
    Returns:
    
        Standard errors.
    """
    def calculate_information(full=True):
        sigma, (m, c) = opt.get_sigma()
        sigma_grad = opt.get_sigma_grad(m, c)
        inv_sigma = chol_inv(sigma)
        sz = len(opt.params)
        info = np.zeros((sz, sz))
        for i in range(sz):
            for k in range(i, sz):
                info[i, k] = np.einsum('ij,ji->', sigma_grad[i] @ inv_sigma,
                                       sigma_grad[k] @ inv_sigma)
        return info + np.triu(info, 1).T if full else info
    try:
        if information == 'expected':
            information = calculate_information()
        elif information == 'observed':
            information = opt.ml_wishart_hessian(opt.params)
    except np.linalg.LinAlgError:
        return [np.nan for _ in range(len(opt.params))]
    asymptoticCov = np.linalg.pinv(information)
    variances = asymptoticCov.diagonal().copy()
    inds = (variances < 0) & (variances > -1e-1)
    variances[inds] = 1e-12
    variances[variances < 0] = np.nan # So numpy won't throw a warning.
    return np.sqrt(variances / (opt.profiles.shape[0] / 2))

def calc_se(opt: Optimizer, information='expected'):
    """Calculates standard errors.
    
    Keyword arguments:
    
        opt         -- Optimizer containing proper parameters' values.
        
        information -- Whether to use "expected" or "observed" Fisher information
                       matrix.
    
    Returns:
    
        Standard errors.
    """
    return calculate_standard_errors(opt, information)

def calculate_z_values(opt: Optimizer, std_errors=None,
                       information='expected'):
    """Calculates z-scores.
    
    Keyword arguments:
    
        opt         -- Optimizer containing proper parameters' values.
        
        std_errors  -- Standard errors in case they were already calculated.
        
        information -- Whether to use expected FIM or observed.
    
    Returns:
   
        Z-scores.
    """
    if std_errors is None:
        std_errors = calculate_standard_errors(opt, information=information)
    return [val / std for val, std in zip(list(opt.params), std_errors)]

def calc_zvals(opt: Optimizer, std_errors=None,
                       information='expected'):
    """Calculates z-scores.
    
    Keyword arguments:
    
        opt         -- Optimizer containing proper parameters' values.
        
        std_errors  -- Standard errors in case they were already calculated.
        
        information -- Whether to use expected FIM or observed.
    
    Returns:
    
        Z-scores.
    """
    return calculate_z_values(opt, std_errors, information)

def calculate_p_values(opt: Optimizer, z_scores=None, information='expected'):
    """Calculates p-values.
    
    Keyword arguments:
    
        opt         -- Optimizer containing proper parameters' values.
        
        z_scores    -- Z-scores in case they were already calculated.
        
        information -- Whether to use expected FIM or observed.
    
    Returns:
    
        P-values.
    """
    if z_scores is None:
        z_scores = calculate_z_values(opt, information=information)
    return [2 * (1 - norm.cdf(abs(z))) for z in z_scores]

def calc_pvals(opt: Optimizer, z_scores=None, information='expected'):
    """Calculates p-values.
    
    Keyword arguments:
    
        opt         -- Optimizer containing proper parameters' values.
        
        z_scores    -- Z-scores in case they were already calculated.
        
        information -- Whether to use expected FIM or observed.
    
    Returns:
    
        P-values.
    """
    return calculate_p_values(opt, z_scores, information)

def gather_statistics(opt: Optimizer, information='expected'):
    """Retrieves all statistics as specified in SEMStatistics structure.
    
    Keyword arguments:
    
        opt         -- Optimizer containing proper parameters' values.
        
        information -- Whether to use expected FIM or observed.
    
    Returns:
    
        SEMStatistics.
    """
    values = opt.params.copy()
    std_errors = calculate_standard_errors(opt, information=information)
    z_scores = calculate_z_values(opt, std_errors, information=information)
    pvalues = calculate_p_values(opt, z_scores, information=information)
    lh = calculate_likelihood(opt)
    aic = calculate_aic(opt, lh)
    bic = calculate_bic(opt, lh)
    paramStats = [ParameterStatistics(val, std, ztest, pvalue)
                  for val, std, ztest, pvalue
                  in zip(values, std_errors, z_scores, pvalues)]
    dof = calculate_dof(opt)
    chi2_base, dof_base = __get_chi2_base(opt)
    chi2 = calculate_chi_square(opt, dof)
    rmsea = calculate_rmsea(opt, chi2[0], dof)
    cfi = calculate_cfi(opt, dof, chi2[0], dof_base, chi2_base)
    gfi = calculate_gfi(opt, chi2[0], chi2_base)
    agfi = calculate_agfi(opt, dof, gfi)
    nfi = calculate_nfi(opt, dof, chi2[0], chi2_base)
    tli = calculate_tli(opt, dof, chi2[0], dof_base, chi2_base)
    return SEMStatistics(dof, lh, opt.last_run[1], chi2, dof_base, chi2_base,
                         rmsea, cfi, gfi, agfi, nfi, tli, aic, bic, paramStats)

