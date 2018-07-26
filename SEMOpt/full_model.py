from .model import Model
from .parser import Parser
from copy import deepcopy
import numpy as np


class FullModel(Model):
    def __init__(self, variable_names: list, model_description: str,
                 ignored_params=set()):
        """
        Key arguments:
        variable_names    -- names of observed variables from Beta matrix.
        model_description -- description of model (optionally empty),
                             assumed to be description of measurement part.
        psi_diags         -- Values to be fixed on Psi diagonal.
        theta_diags       -- Values to be fixed on Theta diagonal.
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
        super().__init__(None, description, force_load=force_load)
