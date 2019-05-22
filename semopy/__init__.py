'''semopy is a Python package that implements numerous SEM-related functonality.'''
from .stats import gather_statistics, calculate_likelihood, calculate_aic,\
                   calculate_bic
from .regsem import StructureAnalyzer
from .visualization import visualize
from .optimizer import Optimizer
from .inspector import inspect
from .model_nl import ModelNL
from .model import Model

name = "semopy"
__version__ = "1.1.7"