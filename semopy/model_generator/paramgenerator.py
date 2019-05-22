'''The module contains methods to generate parameters for a given model.'''
from numpy.random import exponential, uniform
from scipy.stats import truncnorm
from pandas import DataFrame


def trunc_exp():
    t = exponential(1) + 0.1
    if t > 2.2:
        t = uniform(1, 2.2)
    return t


def trunc_norm():
    return truncnorm.rvs(0.1, 2.9)


def generate_parameters(mpart: dict, spart: dict,
                        mpart_generator=trunc_norm,
                        spart_generator=trunc_norm,
                        mpart_fix_value=1.0):
    '''Generates random parameters for the proposed model.
    
    Keyword arguments:
        
        mpart           -- A measurement part.
        
        spart           -- A structural part.
        
        mpart_generator -- A function f() that is used to randomly generate
                           parameters for measurement part.
                           
        spart_generator -- A function f() that is used to randomly generate
                           parameters for structural part.
                           
        mpart_fix_value -- A value to fix with firsts indicators for each latent
                           variable.
                           
    Returns:
        
        Two dictionaries with parameters for spart and mpart in the form
        {'SomeVariable': [(y1, 1.0), (y2, 5.5)]}
    '''
    return generate_parameters_part(mpart, mpart_generator, mpart_fix_value),\
           generate_parameters_part(spart, spart_generator)


def generate_parameters_part(part: dict, generator, fix_first=None):
    d = dict()
    for v, variables in part.items():
        d[v] = list()
        variables = sorted(variables)
        it = iter(variables)
        if fix_first is not None:
            var = next(it)
            d[v].append((var, fix_first))
        for var in it:
            d[v].append((var, generator()))
    return d


def params_set_to_dataframe(mpart: dict, spart: dict, include_first_ind=True):
    '''Translates a set of parameters produced by generate_parameters to a
    pandas' DataFrame.
    
    Keyword arguments:
        
        params -- One of dictionaries returned by generate_parameters.
    Returns:
        
        A pandas' DataFrame.
    '''
    lt = list()
    for lv in spart:
        for rv, est in spart[lv]:
            lt.append({'lhs': lv, 'op': '~', 'rhs': rv, 'est': est})
    for lv in mpart:
        it = iter(mpart[lv])
        if not include_first_ind:
            next(it)
        for rv, est in it:
            lt.append({'lhs': lv, 'op': '=~', 'rhs': rv, 'est': est})
    return DataFrame(lt, columns=['lhs', 'op', 'rhs', 'est'])
    
