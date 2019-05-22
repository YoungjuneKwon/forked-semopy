from .model import Model
from .parser import Parser
from copy import deepcopy
import numpy as np


class FullModel(Model):
    '''FullModel is a model that assumes all variables (except indicators)
    to be connected to each other.
    
    Key arguments:
        
            variable_names    -- names of observed variables from Beta matrix.
            
            model_description -- description of model (optionally empty),
                                 assumed to be description of measurement part.
                                 
            fix_theta         -- Assume Theta to be an identity block matrix.
            
    '''
    def __init__(self, variable_names: list, model_description: str,
                 ignored_params=set(), fix_theta=False):
        """
        Key arguments:
            
            variable_names    -- names of observed variables from Beta matrix.
            
            model_description -- description of model (optionally empty),
                                 assumed to be description of measurement part.
                                 
            fix_theta         -- Assume Theta to be an identity block matrix.
        """
        ops = Model.operations
        parser = Parser(ops)
        description = parser.parse(model_description)
        latents = [v for v in description if description[v][ops.MEASUREMENT]]
        force_load = set(variable_names)
        variable_names = sorted(latents) + variable_names
        for i, j in np.nditer(np.tril_indices(len(variable_names), -1)):
            lval, rval = variable_names[i], variable_names[j]
            if (lval, rval) not in ignored_params:
                description[lval][ops.REGRESSION][rval]
        self.description = deepcopy(description)
        if fix_theta:
            inds = [ind for lv in latents
                    for ind in description[lv][ops.MEASUREMENT]]
            for ind in inds:
                description[ind][ops.COVARIANCE][ind] = [1.0]
        super().__init__(None, description, force_load=force_load)
