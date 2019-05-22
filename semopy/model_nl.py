'''This module contains a WIP-version of nonlinear SEM model.'''
from itertools import combinations, product
from .model import Model
from enum import Enum
import pandas as pd


class SEMOperationsNL(Enum):
    REGRESSION = '~'
    MEASUREMENT = '=~'
    COVARIANCE = '~~'
    TYPE = 'is'
    NLREGRESSION = '@~'

class ModelNL(Model):
    operations = SEMOperationsNL

    def process_description(self, description, force_load=set()):
        """Classifies variables and performns necessary initial operations over
        model.

        Key arguments:
            
            description -- A dictionary structure returned by Parser.parse().
            
            force_load  -- Observed variables that may not be necessary
                           provided in description yet must be taken in
                           account.
        """
#       Taking care of nonlinear inferences.
        ops = self.operations
        lvs = [v for v in description if description[v][ops.NLREGRESSION]]
        for lv in lvs:
            if description[lv][ops.NLREGRESSION]:
                t = {v: list() for v in description[lv][ops.NLREGRESSION]
                     if 'n' not in description[lv][ops.NLREGRESSION][v]}
                if description[lv][ops.REGRESSION]:
                    t = {v: list() for v in t
                         if v not in description[lv][ops.REGRESSION]}
                    description[lv][ops.REGRESSION].update(t)
                else:
                    description[lv][ops.REGRESSION] = t.copy()
                for a, b in combinations(description[lv][ops.NLREGRESSION], 2):
                    if a > b:
                        a, b = b, a
                    name = '{}*{}'.format(a, b)
                    description[lv][ops.REGRESSION][name] = list()
                    a_inds = description[a][ops.MEASUREMENT]
                    b_inds = description[b][ops.MEASUREMENT]
                    if a_inds and not b_inds or not a_inds and b_inds:
                        raise Exception("Nonlinear interactions between\
                                        latent and observed variables are not\
                                        supported.")
                    for a_ind, b_ind in product(a_inds, b_inds):
                        name_ind = '{}*{}'.format(a_ind, b_ind)
                        description[name][ops.MEASUREMENT][name_ind] = list()                                             
        super().process_description(description, force_load)

    def classify_variables(self, description, force_load=set()):
        """Classifies variables and performns necessary initial operations over
        model.

        Key arguments:
            
            description -- A dictionary structure returned by Parser.parse().
            
            force_load  -- Observed variables that may not be necessary
                           provided in description yet must be taken in
                           account.
        """
        super().classify_variables(description, force_load)
        self.vars['NLterms'] = [v for v in self.vars['All'] if '*' in v]

    def load_dataset(self, data: pd.DataFrame, center=True, ordcor=True):
        """Loads dataset and applies starting values.
        
        Keyword arguments:
            
            data   -- A Pandas' DataFrame containing data.
            
            center -- Center data taking in account categorical variables
                      if ordcor is True.
                      
            ordcor -- Whether to use compute a heterogenous correlation matrix
                      instead of covariance matrix if categorical variables are
                      present.
        """
        data = data.copy()
        for term in self.vars['NLterms']:
            if term not in self.vars['Latents']:
                a, b = term.split('*')
                data[term] = data[a] * data[b]
        super().load_dataset(data, center, ordcor)
