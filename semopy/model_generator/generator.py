'''The module contains an umbrella methods that incroporate in themselves
methods from other modules of model_generator for simplicity purposes.
Less options, but much easier to use.'''
from .structgenerator import generate_measurement_part,\
                             generate_structural_part, create_model_description
from .datagenerator import generate_data
from .paramgenerator import generate_parameters, params_set_to_dataframe
from ..parser import ExampleOperations as EO
from collections import defaultdict
from scipy.stats import uniform
from numpy.random import choice
from functools import partial
from ..parser import Parser


def param_gen(scale, shift):
    v = scale * uniform.rvs(0.1, 1.1)
    if choice([True, False]):
        return v + shift
    else:
        return -v + shift

def generate_model(n_obs: int, n_lat: int, n_manif: tuple, p_manif: float,
                   n_cycles: int, scale: float, n_size: int, f_gen=None):
    """Generates a measurement, structural parts, parameters and data. Bear in
    mind that this method is a wrapper around generatte_measurement_part,
    generated_structura_part, generate_parameters and generate_data. The
    functionality embodied in those methods is much broader than in this one.
    If one wishes to approach the generation process with a greater
    versatility, consider using those methods instead (see their __doc__
    strings). This methods exists solely for simplicity purposes.
    Keyword arguments:
        n_obs    -- A number of observed variables in structural part.
        n_lat    -- A number of latent variables in structural part.
        n_manif  -- A range from which a number of manifest variables will be
                    chosen for each latent variable.
        p_manif  -- A percentage of manifest variables that will be merged
                    together.
        n_cycles -- A number of cycles to be present in the model.
        scale    -- All parameters sampled from uniform distribution on interval
                    [-1, -0.1]u[0.1, 1] are multiplied by this value.
        n_size   -- A number of data samples.
        f_gen    -- A generator function from where parameters' values are 
                    sampled. Overwrites scale parameter making it useless.
                    Leave none to use default distribution settings.
    Returns:
        Model description, DataFrame with parameters values, data.
    """
    global param_gen
    if f_gen is None:
        f_gen = partial(param_gen, scale, 0)
    mpart = generate_measurement_part(n_lat, n_manif, p_manif)
    spart, tm = generate_structural_part(mpart, n_obs, num_cycles=n_cycles)
    params_mpart, params_spart = generate_parameters(mpart, spart,
                                                     mpart_generator=f_gen,
                                                     spart_generator=f_gen)
    data = generate_data(mpart, spart, params_mpart, params_spart,
                         n_size, tm)
    params_df = params_set_to_dataframe(params_mpart, params_spart, False)
    model = create_model_description(mpart, spart)
    return model, params_df, data

def get_parts_from_description(desc: str):
    """Retrieves m_part and s_part for generator functions given string
    description of the model.
    Keyword arguments:
        desc -- A description of model.
    Returns:
        m_part and s_part.
    """
    m = Parser()
    desc = m.parse(desc)
    mpart, spart = defaultdict(set), defaultdict(set)
    for lv, ops in desc.items():
        for op, rvs in ops.items():
            if rvs:
                if op is EO.MEASUREMENT:
                    mpart[lv].update(set(rvs.keys()))
                elif op is EO.REGRESSION:
                    spart[lv].update(set(rvs.keys()))
    return mpart, spart
    
def generate_params_for_model(model: str, scale=1.0, f_gen=None):
    """Generates parameters for model in a form required for generator
    functions given model description.
    Keyword arguments:
        model -- A model description.
        scale -- A scale paramater as in generate_model function.
        f_gen -- An f_gen parameter as in generate_model function.
    Returns:
        (params_mpart, params_spart), a dataframe of parameters.
    """
    global param_gen
    mpart, spart = get_parts_from_description(model)
    if f_gen is None:
        f_gen = partial(param_gen, scale, 0)
    params_mpart, params_spart = generate_parameters(mpart, spart,
                                                     mpart_generator=f_gen,
                                                     spart_generator=f_gen)
    params_df = params_set_to_dataframe(params_mpart, params_spart, False)
    return (params_mpart, params_spart), params_df
    

def generate_data_for_model(model: str, n_samples: int, params=None,
                            f_gen=None):
    """Generates a dataset for given model.
    Keyword arguments:
        model     -- A model description.
        n_samples -- A number of samples in a dataset.
        params    -- Parameters in a form (params_mpart, params_spart).
        f_gen     -- A generator function from where random data is sampled.
                     Leave None to use default distribution settings.
    Returns:
        A Pandas DataFrame containing dataset for a model.
    """
    parts = get_parts_from_description(model)
    if params is None:
        params = generate_params_for_model(model)[0]
    return generate_data(*parts, *params, n_samples, None)