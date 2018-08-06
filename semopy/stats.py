from .optimizer import Optimizer
from collections import namedtuple
from scipy.special import multigammaln
from scipy.stats import norm
from .utils import chol_inv
import numpy as np

ParameterStatistics = namedtuple('ParametersStatistics',
                                 ['value', 'se', 'zscore', 'pvalue'])
SEMStatistics = namedtuple('SEMStatistics', ['dof', 'ml', 'aic', 'bic',
                                             'params'])


def calculate_dof(opt: Optimizer):
    """Calculates degrees of freedom.

    Keyword arguments:
    opt -- Optimizer containing proper parameters' values.

    Returns:
    Degrees of freedom.
    """
    p = opt.mx_cov.shape[0]  # Num of observed variables.
    num_cov = p * (p + 1) / 2
    return num_cov - len(opt.params)


def calculate_likelihood(opt: Optimizer):
    """Calculates Wishart likelihood.

    Keyword arguments:
    opt -- Optimizer containing proper parameters' values.

    Returns:
    Wishart likelihood.
    """
    sigma, _ = opt.get_sigma()
    p = sigma.shape[0]
    n = opt.model.num_params
    tr = -np.einsum('ij,ji->', opt.mx_cov, chol_inv(sigma)) / 2
    det_s = (n - p - 1) / 2 * opt.cov_logdet
    det_sigma = - n * np.linalg.slogdet(sigma)[1] / 2
    det = det_s + det_sigma
    c = - n / 2 * np.log(2)
    lngamma = -multigammaln(n / 2, p)
    return tr + det + c + lngamma


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


def calculate_bic(opt: Optimizer, lh=None):
    """Calculates BIC.

    Keyword arguments:
    opt -- Optimizer containing proper parameters' values.
    lh  -- Likelihood in case it was already calculated.

    Returns:
    BIC.
    """
    if lh is None:
        lh = opt.ml_wishart(opt.params)
    k, n = len(opt.params), opt.profiles.shape[0]
    return np.log(n) * k - 2 * lh


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
    if information == 'expected':
        information = calculate_information()
    elif information == 'observed':
        information = opt.ml_wishart_hessian(opt.params)
    asymptoticCov = np.linalg.pinv(information)
    variances = asymptoticCov.diagonal().copy()
    inds = (variances < 0) & (variances > -1e-1)
    variances[inds] = 1e-12
    return np.sqrt(variances / (opt.profiles.shape[0] / 2))


def calculate_z_values(opt: Optimizer, std_errors=None):
    """Calculates z-scores.

    Keyword arguments:
    opt        -- Optimizer containing proper parameters' values.
    std_errors -- Standard errors in case they were already calculated.

    Returns:
    Z-scores.
    """
    if std_errors is None:
        std_errors = calculate_standard_errors(opt)
    return [val / std for val, std in zip(list(opt.params), std_errors)]


def calculate_p_values(opt: Optimizer, z_scores=None):
    """Calculates p-values.

    Keyword arguments:
    opt      -- Optimizer containing proper parameters' values.
    z_scores -- Z-scores in case they were already calculated.

    Returns:
    P-values.
    """
    if z_scores is None:
        z_scores = calculate_z_values(opt)
    return [2 * (1 - norm.cdf(abs(z))) for z in z_scores]


def gather_statistics(opt: Optimizer):
    """Retrieves all statistics as specified in SEMStatistics structure.

    Keyword arguments:
    opt      -- Optimizer containing proper parameters' values.

    Returns:
    SEMStatistics.
    """
    values = opt.params.copy()
    std_errors = calculate_standard_errors(opt)
    z_scores = calculate_z_values(opt, std_errors)
    pvalues = calculate_p_values(opt, z_scores)
    lh = calculate_likelihood(opt)
    aic = calculate_aic(opt, lh)
    bic = calculate_bic(opt, lh)
    paramStats = [ParameterStatistics(val, std, ztest, pvalue)
                  for val, std, ztest, pvalue
                  in zip(values, std_errors, z_scores, pvalues)]
    dof = calculate_dof(opt)
    return SEMStatistics(dof, lh, aic, bic, paramStats)
