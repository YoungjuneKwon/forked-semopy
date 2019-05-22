'''Inspector module is responsible for providing a user information on
parameters' estimatees in a friendly manner.'''
from .stats import gather_statistics
from .optimizer import Optimizer
from sys import stdout
import pandas as pd
import numpy as np
import sys


def inspector_get_mx(opt: Optimizer, mx_name: str):
    """Get matrix as a DataFrame.

    Keyword arguments:
        
        opt     -- SEM Optimizer.
        
        mx_name -- Matrix's name.

    Returns:
        
        Pandas DataFrame.
    """
    if mx_name == 'Beta':
        mx = opt.mx_beta
        rows, cols = opt.model.beta_names
    elif mx_name == 'Lambda':
        mx = opt.mx_lambda
        rows, cols = opt.model.lambda_names
    elif mx_name == 'Psi':
        mx = opt.mx_psi
        rows, cols = opt.model.psi_names
    elif mx_name == 'Theta':
        mx = opt.mx_theta
        rows, cols = opt.model.theta_names
    elif mx_name == 'Sigma':
        mx = opt.model.calculate_sigma(opt.mx_beta, opt.mx_lambda, opt.mx_psi,
                                       opt.mx_theta)
        mx = mx[0]
        rows, cols = opt.model.theta_names
    elif mx_name == 'Cov' or mx_name == 'S':
        mx = opt.mx_cov
        rows, cols = opt.model.theta_names
    else:
        raise Exception("Unkown matrix {}".format(mx_name))
    return pd.DataFrame(data=mx, index=rows, columns=cols)


def inspect_list(opt: Optimizer):
    """Returns a DataFrame table containing all parameters in models in a
    clear, readable from. Statistics provided as well.

    Keyword arguments:
        
        opt -- A SEM optimizer.

    Returns:
        
        Pandas' DataFrame.
    """
    stats = gather_statistics(opt).params
    lt = list()
    for i in range(*opt.model.beta_range):
        n, m = opt.model.parameters['Beta'][i - opt.model.beta_range[0]]
        rows, cols = opt.model.beta_names
        lval, rval = rows[n], cols[m]
        lt.append({'lval': lval, 'op': '~', 'rval': rval,
                   'Value': stats[i].value, 'SE': stats[i].se,
                   'Z-score': stats[i].zscore, 'P-value': stats[i].pvalue})
    for i in range(*opt.model.lambda_range):
        n, m = opt.model.parameters['Lambda'][i - opt.model.lambda_range[0]]
        rows, cols = opt.model.lambda_names
        lval, rval = rows[n], cols[m]
        lt.append({'lval': lval, 'op': '=~', 'rval': rval,
                   'Value': stats[i].value, 'SE': stats[i].se,
                   'Z-score': stats[i].zscore, 'P-value': stats[i].pvalue})
    for i in range(*opt.model.psi_range):
        n, m = opt.model.parameters['Psi'][i - opt.model.psi_range[0]]
        rows, cols = opt.model.psi_names
        lval, rval = rows[n], cols[m]
        lt.append({'lval': lval, 'op': '~~', 'rval': rval,
                   'Value': stats[i].value, 'SE': stats[i].se,
                   'Z-score': stats[i].zscore, 'P-value': stats[i].pvalue})
    for i in range(*opt.model.theta_range):
        n, m = opt.model.parameters['Theta'][i - opt.model.theta_range[0]]
        rows, cols = opt.model.theta_names
        lval, rval = rows[n], cols[m]
        lt.append({'lval': lval, 'op': '~~', 'rval': rval,
                   'Value': stats[i].value, 'SE': stats[i].se,
                   'Z-score': stats[i].zscore, 'P-value': stats[i].pvalue})
    df = pd.DataFrame(lt, columns=['lval', 'op', 'rval', 'Value', 'SE',
                                   'Z-score', 'P-value'])
    return df.sort_values(['op', 'lval', 'rval'])


def inspect(opt: Optimizer, mode='list', what='est', output=stdout):
    """Outputs all the information available given SEM optimizer with
    appropriate parameters.

    Keyword arguments:
        opt     -- SEM Optimizer.
        
        mode    -- 'mx' or 'list', inspect prints matrices or list of operations
                    retrieved by inspect_list.
                    
        what    -- 'est' or 'start', if 'est' then estimated parameters are
                    applied, if 'start' then starting parameters are applied.
                    
        output  -- Output stream.

    Returns:
        
        A list of of pairs (matrix_name, df matrix) if mode equals 'mx' or
        pandas DataFrame containing operations if mode=='list'.
    """
    ret = None
    np.set_printoptions(threshold=sys.maxsize)
    tmp_params = opt.params.copy()
    if what == 'est':
        pass
    elif what == 'start':
        opt.params = opt.starting_params
    else:
        raise Exception("Unkown 'what' parameter for inspection.")
    opt.update_matrices(opt.params)
    if mode == 'mx':
        ret = list()
        matrices_to_print = ('Beta', 'Lambda', 'Psi', 'Theta', 'Sigma', 'Cov')
        for mx_name in matrices_to_print:
            mx = inspector_get_mx(opt, mx_name)
            ret.append((mx_name, mx))
            print("{}:".format(mx_name), file=output)
            with pd.option_context('display.max_rows', None,
                                   'display.max_columns', None):
                print(mx, file=output)
    elif mode == 'list':
        ret = inspect_list(opt)
    else:
        raise Exception("Unkown 'mode' parameter for inspection")
    opt.params = tmp_params
    return ret
