from abc import ABC, abstractmethod
from pandas import DataFrame
from .parser import Parser
from numpy import unique
from .utils import cov
from enum import Enum


class SEMOperations(Enum):
    REGRESSION = '~'
    MEASUREMENT = '=~'
    COVARIANCE = '~~'


class ModelABC(ABC):
    operations = SEMOperations

    @abstractmethod
    def __init__(self, model_description: str, description=None,
                 force_load=set()):
        """
        Key arguments:
        model_description -- A string containing SEM model description in
                             a valid syntax.
        description       -- A parsed model_description in the form returned by
                             Parser.parse().
        force_load        -- Observed variables that may not be necessary
                             provided in description yet must be taken in
                             account.
        """
        if description is None:
            parser = Parser(ModelABC.operations)
            description = parser.parse(model_description)
        self.param_vals = None
        self.process_description(description, force_load)

    def process_description(self, description, force_load=set()):
        """Classifies variables and performns necessary initial operations over
        model.

        Key arguments:
        description -- A dictionary structure returned by Parser.parse().
        force_load  -- Observed variables that may not be necessary
                       provided in description yet must be taken in
                       account.
        """
        self.classify_variables(description, force_load)
        self.prepare_parameters(description)

    def classify_variables(self, description, force_load=set()):
        """Classifies variables and performns necessary initial operations over
        model.

        Key arguments:
        description -- A dictionary structure returned by Parser.parse().
        force_load  -- Observed variables that may not be necessary
                       provided in description yet must be taken in
                       account.
         """
        self.vars = dict()
        in_arrows = set()
        out_arrows = set()
        latents = set()
        indicators = set()
        ops = ModelABC.operations
        for lv in description:
            rvals_regression = description[lv][ops.REGRESSION].keys()
            if rvals_regression:
                in_arrows.add(lv)
                out_arrows.update(rvals_regression)
            inds = description[lv][ops.MEASUREMENT].keys()
            if inds:
                latents.add(lv)
                indicators.update(inds)
        spart = in_arrows | out_arrows | latents
        force_load = {v for v in force_load if v not in spart}
        spart = spart | force_load
        exogenous = spart - in_arrows
        endogenous = spart - exogenous
        outputs = in_arrows - out_arrows
        mpart = latents | indicators
        observed = spart - latents

        self.vars['LatExo'] = sorted(list(latents & exogenous))
        self.vars['LatEndo'] = sorted(list(latents & endogenous))
        self.vars['Latents'] = sorted(list(latents))
        self.vars['ObsExo'] = sorted(list(observed & exogenous))
        self.vars['ObsEndo'] = sorted(list(observed & endogenous))
        self.vars['Observed'] = self.vars['ObsEndo'] + self.vars['ObsExo']
        self.vars['Indicators'] = sorted(list(indicators))
        self.vars['Outputs'] = sorted(list(outputs))
        self.vars['SPart'] = self.vars['Latents'] + self.vars['Observed']
        self.vars['MPart'] = sorted(list(mpart))
        self.vars['IndsObs'] = self.vars['Indicators'] + self.vars['Observed']
        self.vars['All'] = self.vars['Indicators'] + self.vars['Latents'] +\
                           self.vars['Observed']
        self.vars['Binary'] = list()
        self.vars['Categorical'] = list()

    def load_dataset(self, data: DataFrame, bias=True):
        data = data[self.vars['IndsObs']]
        for v in self.vars['IndsObs']:
            num_uniqs = len(unique(data[v]))
            if num_uniqs == 2:
                self.vars['Binary'].append(v)
            elif num_uniqs < data.shape[0] / 10:
                if num_uniqs == 1:
                    print("Warning: variable {} attains only one value".format(v))
                self.vars['Categorical'].append(v)
        self.raw_data = data.values
        self.mx_cov = cov(self.raw_data, bias=bias)

    @abstractmethod
    def parse_operation(self, op, lvalue, rvalue, args):
        """Creates a parameter or performs necessary operations given opcode.

        Keyword arguments:
        op     -- An operation's code.
        lvalue -- A left value.
        rvalue -- A right value.
        args   -- Arguments of the operation.
        """
        pass

    @abstractmethod
    def prepare_parameters(self, description):
        """Creates a dict of parameters and performns necessary initial
        operations over model.

        Keyword arguments:
        description -- A structure returned by Parser.
        """
        for lv in description:
            for op in description[lv]:
                for rv, args in description[lv][op].items():
                    self.parse_operation(op, lv, rv, args)
