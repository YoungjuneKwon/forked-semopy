'''The module contains methods for generating a data for a given model.'''

from .utils import get_tuple_index, ThreadsManager
from numpy.random import normal, uniform
from functools import partial
from pandas import DataFrame


DEFAULT_NORMAL = partial(uniform, -1, 3.4641)
DEFAULT_ERROR = partial(normal, 0, 1)


def generate_data(mpart: dict, spart: dict, mpart_params: dict,
                  spart_params: dict, num_rows: int,
                  threads: ThreadsManager,
                  mpart_generator=DEFAULT_NORMAL,
                  spart_generator=DEFAULT_NORMAL,
                  error_generator=DEFAULT_ERROR):
    '''Generates a datasample given the model and it's parameters.
    
    Keyword arguments:
        mpart           -- A measurement part.
        
        spart           -- A structural part.
        
        params_mpart    -- Measurement part parameters.
        
        params_spart    -- Structural part parameters.
        
        num_rows        -- Number of samples in a dataset to be generated.
        
        threads         -- An auxilary object produced by generate_structural_part.
        
        mpart_generator -- A function f(shape) that is used to randomly generate
                           data for measurement part.
                           
        spart_generator -- A function f(shape) that is used to randomly generate
                           data for structural part.
                           
        error_generator -- A function f(shape) that is used to randomly generate
                           errors for data.
                           
    Returns:
        
        A dataframe table.
    '''
    latents = set(mpart.keys())
    indicators = {ind for lv in latents for ind in mpart[lv]}
    out_arrows = {mf for v in spart for mf in spart[v]}
    in_arrows = {v for v in spart}
    exogenous = out_arrows - in_arrows
    spart_vars = in_arrows | out_arrows
    variables = indicators | spart_vars
    if threads is None:
        threads = ThreadsManager()
        threads.load_from_dict(spart, True)
    threads.load_from_dict(mpart)
    data = DataFrame(0.0, index=range(num_rows),
                     columns=sorted(list(variables)))
    # "Filling" latent variables with data using their respective indicators.
#    for lv, indicators in mpart.items():
#        lv_params = mpart_params[lv]
#        for indicator in indicators:
#            data[indicator] = mpart_generator([num_rows]) +\
#                              error_generator([num_rows])
#            i = get_tuple_index(lv_params, 0, indicator)
#            mult = 1 / lv_params[i][1]
#            data[lv] += mult * data[indicator]
    for v in exogenous:
        data[v] = spart_generator([num_rows])
    data_copy = data.copy()
    for v in exogenous:
        data_ref = data_copy.copy()
        it = iter(threads.get_confluent_path(v))
        prev = next(it)
        for nodes in it:
            visited = set()
            for a, b in zip(nodes, prev):
                if (a, b) not in visited and a is not None:
                    if a not in indicators:
                        p = spart_params[a]
                        i = get_tuple_index(p, 0, b)
                    else:
                        p = mpart_params[b]
                        i = get_tuple_index(p, 0, a)
                    mult = p[i][1]
                    t = mult * (data_ref[b] + error_generator([num_rows]))
                    data_ref[a] += t
                    data[a] += t
                visited.add((a, b))
            prev = nodes
    return data.drop(latents, 1)