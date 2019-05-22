'''This module contains Model class that is used a "meat and bones" of a SEM\\ 
model.'''
from itertools import combinations, product, chain
from .model_abstract import ModelABC
from scipy.stats import linregress
from pandas import DataFrame
import numpy as np


class Model(ModelABC):
    '''Model is responsible for building a mathematical backbone of the
    SEM-model given a textual (string) description and setting up initial
    values from data.
    
    Keyword arguments:
        
            model_description -- A text description of model in semopy syntax.
            
            description       -- A description in a dictionary form as returned
                                 by Parser.
                                 
            force_load        -- A set of variables that may not be present in
                                 the description, yet should be represented in
                                 a model at least as isolated exogenous
                                 variables.
                                 
            baseline          -- Should model be initialized in a baseline mode
                                 (all parameters in Beta and Lambda are
                                  fixed and set to 0 and 1 respectively).
                                 
            psi_mode          -- "Full" for default Psi matrix.
                                 "Diag" for default Psi matrix with zero
                                 covariances.
                                 "DiagFixed" for Psi matrix with all diagonal
                                 elements fixed to their sample values.
                                 "DiagParam" for Psi matrix with all diagonal
                                 elements used as parameters.
    '''
    def __init__(self, model_description: str, description=None,
                 force_load=set(), baseline=False, psi_mode='Full'):
        """Creates a model instance.

        Keyword arguments:
            model_description -- A text description of model in semopy syntax.
            description       -- A description in a dictionary form as returned
                                 by Parser.
            force_load        -- A set of variables that may not be present in
                                 the description, yet should be represented in
                                 a model at least as isolated exogenous
                                 variables.
            baseline          -- Should model be initialized in a baseline mode
                                 (all parameters in Beta and Lambda are
                                  fixed and set to 0 and 1 respectively)?
            psi_mode          -- "Full" for default Psi matrix.
                                 "Diag" for default Psi matrix with zero
                                 covariances.
                                 "DiagFixed" for Psi matrix with all diagonal
                                 elements fixed to their sample values.
                                 "DiagParam" for Psi matrix with all diagonal
                                 elements used as parameters.
        """
        self.psi_mode = psi_mode
        self.baseline = baseline
        super().__init__(model_description, description, force_load)
        self.prepare_diff_matrices()

    def prepare_parameters(self, description):
        self.mx_beta, self.beta_names = self.build_beta()
        self.mx_lambda, self.lambda_names = self.build_lambda()
        self.mx_psi, self.psi_names = self.build_psi()
        self.mx_theta, self.theta_names = self.build_theta()
        # Order is important.
        self.mx_names = ('Beta', 'Lambda', 'Psi', 'Theta')
        self.mx_names_symm = ('Psi', 'Theta')
        self.matrices = {'Beta': self.mx_beta, 'Lambda': self.mx_lambda,
                         'Psi': self.mx_psi, 'Theta': self.mx_theta}
        self.parameters = {name: list() for name in self.mx_names}
        self.first_indicators = {lv: str() for lv in self.vars['Latents']}
        self.fixed_covars = list()
        self.num_params = 0
        ops = self.operations
        if self.psi_mode == 'Full':
            for v1, v2 in chain(combinations(self.vars['LatExo'], 2),
                                combinations(self.vars['Outputs'], 2)):
                if v1 not in description[v2][ops.COVARIANCE] and\
                   v2 not in description[v1][ops.COVARIANCE]:
                    description[v1][ops.COVARIANCE][v2] = list()
        if self.psi_mode == 'Full' or self.psi_mode == 'Diag':
            it = chain(self.vars['ObsEndo'], self.vars['Latents'],
                       self.vars['Indicators'])
        elif self.psi_mode == 'DiagFixed':
            it = self.vars['Indicators']
        elif self.psi_mode == 'DiagParams':
            it = chain(self.vars['SPart'], self.vars['Indicators'])
        for v in it:
            if v not in description[v][ops.COVARIANCE]:
                description[v][ops.COVARIANCE][v] = list()                             
                    
        super().prepare_parameters(description)
        for mx in self.parameters:
            self.parameters[mx] = sorted(self.parameters[mx])
        for v in self.vars['Observed']:
            i, j = self.lambda_names[0].index(v), self.lambda_names[1].index(v)
            self.mx_lambda[i, j] = 1.0

    def load_dataset(self, data: DataFrame, center=True, ordcor=False):
        """Loads dataset and applies starting values.
        
        Keyword arguments:
            
            data   -- A Pandas' DataFrame containing data.
            
            center -- Center data taking in account categorical variables
                      if ordcor is True.
                      
            ordcor -- Whether to use compute a heterogenous correlation matrix
                      instead of covariance matrix if categorical variables are
                      present. If iterable, then elements in ordcor are assumed
                      to be ordinal and no ordinality tests are run.
        """
        super().load_dataset(data, center, ordcor)
        cov = DataFrame(self.mx_cov, index=self.vars['IndsObs'],
                        columns=self.vars['IndsObs'])
        if self.psi_mode == 'Full':
            for v1, v2 in chain(product(self.vars['ObsExo'], repeat=2),
                                self.fixed_covars):
                ind = (self.psi_names[0].index(v1),
                       self.psi_names[1].index(v2))
                self.mx_psi[ind] = cov[v1][v2]
                self.mx_psi[ind[::-1]] = self.mx_psi[ind]
        elif self.psi_mode == 'Diag':
            for v in self.vars['ObsExo']:
                i = self.psi_names[0].index(v)
                self.mx_psi[i, i] = cov[v][v]
        elif self.psi_mode == 'DiagFixed':
            for v in self.vars['Observed']:
                i = self.psi_names[0].index(v)
                self.mx_psi[i, i] = cov[v][v]
            # TODO: add Theta support!
        self.postprocess_parameters(data, cov)

    def apply_parameters(self, params: np.array, mx_beta: np.array,
                         mx_lambda: np.array, mx_psi: np.array,
                         mx_theta: np.array):
        i = self.beta_params_inds
        mx_beta[i[:, 0], i[:, 1]] = params[self.beta_range[0]:self.beta_range[1]]
        i = self.lambda_params_inds
        mx_lambda[i[:, 0], i[:, 1]] = params[self.lambda_range[0]:self.lambda_range[1]]
        i = self.psi_params_inds
        mx_psi[i[:, 0], i[:, 1]] = params[self.psi_range[0]:self.psi_range[1]]
        i = self.psi_params_inds_t
        mx_psi[i[:, 0], i[:, 1]] = params[self.psi_range[0]:self.psi_range[1]]
        i = self.theta_params_inds
        mx_theta[i[:, 0], i[:, 1]] = params[self.theta_range[0]:self.theta_range[1]]
        i = self.theta_params_inds_t
        mx_theta[i[:, 0], i[:, 1]] = params[self.theta_range[0]:self.theta_range[1]]

    def postprocess_parameters(self, data: DataFrame, cov: DataFrame):
        """Creates a fast-to-use array of parameters.

        Keyword arguments:
            
            data -- A dataframe with sample data.
            
            cov  -- A variance-covariance matrix.
        """
        self.param_vals = np.zeros(self.num_params)
        params_inds = np.zeros((0,), dtype=int)
        k = 0
        self.beta_range = k
        for ind in self.parameters['Beta']:
            params_inds = np.append(params_inds, ind)
            if np.isnan(self.mx_beta[ind]):
                self.mx_beta[ind] = 0
            self.param_vals[k] = self.mx_beta[ind]
            k += 1
        self.beta_range = (self.beta_range, k)
        self.lambda_range = k
        for ind in self.parameters['Lambda']:
            params_inds = np.append(params_inds, ind)
            if np.isnan(self.mx_lambda[ind]):
                lambda_names = self.lambda_names
                l, r = lambda_names[0][ind[0]], lambda_names[1][ind[1]]
                r = self.first_indicators[r]
                self.mx_lambda[ind] = linregress(data[r], data[l]).slope
            self.param_vals[k] = self.mx_lambda[ind]
            k += 1
        self.lambda_range = (self.lambda_range, k)
        self.psi_range = k
        for ind in self.parameters['Psi']:
            params_inds = np.append(params_inds, ind)
            if np.isnan(self.mx_psi[ind]):
                l, r = self.psi_names[0][ind[0]], self.psi_names[1][ind[1]]
                if ind[0] == ind[1]:
                    if l in self.vars['Latents']:
                        self.mx_psi[ind] = 0.05
                    else:
                        
                        self.mx_psi[ind] = cov[l][r] / 2.0
                else:
                    self.mx_psi[ind] = 0.0
                    self.mx_psi[ind[::-1]] = 0.0
                    # TODO: check if cov instead of zero in case of observable
                    # variables works better.
            self.mx_psi[ind[::-1]] = self.mx_psi[ind]
            self.param_vals[k] = self.mx_psi[ind]
            k += 1
        self.psi_range = (self.psi_range, k)
        self.theta_range = k
        for ind in self.parameters['Theta']:
            l, r = self.theta_names[0][ind[0]], self.theta_names[1][ind[1]]
            params_inds = np.append(params_inds, ind)
            if l == r:
                self.mx_theta[ind] = cov[l][r] / 2
            else:
                self.mx_theta[ind] = 0.0
                self.mx_theta[ind[::-1]] = 0.0
#                self.mx_theta[ind[::-1]] = self.mx_theta[ind]
            self.param_vals[k] = self.mx_theta[ind]
            k += 1
        self.theta_range = (self.theta_range, k)
        params_inds = np.reshape(params_inds, (k, 2))
        beta_range, lmb_range = self.beta_range, self.lambda_range
        psi_range, theta_range = self.psi_range, self.theta_range
        self.beta_params_inds = params_inds[beta_range[0]:beta_range[1], :]
        self.lambda_params_inds = params_inds[lmb_range[0]:lmb_range[1], :]
        self.psi_params_inds = params_inds[psi_range[0]:psi_range[1], :]
        self.psi_params_inds_t = self.psi_params_inds[:, ::-1]
        self.theta_params_inds = params_inds[theta_range[0]:theta_range[1], :]
        self.theta_params_inds_t = self.theta_params_inds[:, ::-1]

    def parse_operation(self, op, lvalue, rvalue, args):
        super().parse_operation(op, lvalue, rvalue, args)
        ops = self.operations
        if op == ops.REGRESSION:
            ind_lv = self.beta_names[0].index(lvalue)
            ind_rv = self.beta_names[1].index(rvalue)
            ind = (ind_lv, ind_rv)
            if ind in self.parameters['Beta']:
                raise Exception("Operation {} {} {} already specified.".format(lvalue, op, rvalue))
            try:
                if self.baseline:
                    self.mx_beta[ind] = 0
                else:
                    self.mx_beta[ind] = float(args[0])
            except Exception:
                self.num_params += 1
                self.parameters['Beta'].append(ind)
                self.mx_beta[ind] = np.nan
        elif op == ops.MEASUREMENT:
            ind_lv = self.lambda_names[1].index(lvalue)
            ind_rv = self.lambda_names[0].index(rvalue)
            ind = (ind_rv, ind_lv)
            if ind in self.parameters['Lambda']:
                raise Exception("Operation {} {} {} already specified.".format(lvalue, op, rvalue))
            try:
                if self.baseline:
                    self.mx_lambda[ind] = 1.0
                else:
                    self.mx_lambda[ind] = float(args[0])
                    if not self.first_indicators[lvalue]:
                        self.first_indicators[lvalue] = rvalue
            except Exception:
                # Let's make sure that we didn't fix at least one variable yet.
                if np.any(self.mx_lambda[:, ind[1]] != 0.0):
                    self.num_params += 1
                    self.parameters['Lambda'].append(ind)
                    self.mx_lambda[ind] = np.nan
                else:
                    self.mx_lambda[ind] = 1.0
                    self.first_indicators[lvalue] = rvalue
        elif op == ops.COVARIANCE:
            try:
                ind_lv = self.psi_names[0].index(lvalue)
                ind_rv = self.psi_names[1].index(rvalue)
                mx_name = 'Psi'
                mx = self.mx_psi
            except Exception:
                ind_lv = self.theta_names[0].index(lvalue)
                ind_rv = self.theta_names[1].index(rvalue)
                mx_name = 'Theta'
                mx = self.mx_theta
            ind = (ind_lv, ind_rv)
            if ind in self.parameters[mx_name]:
                raise Exception("Operation {} {} {} already specified.".format(lvalue, op, rvalue))
            try:
                if self.baseline and (ind_lv != ind_rv or\
                                     lvalue in self.vars['Latents']):
                    mx[ind] = 0
                else:
                    mx[ind] = float(args[0])
            except Exception:
                if len(args) and args[0] == 'fixcv':
                    self.fixed_covars.append((lvalue, rvalue))
                else:
                    self.num_params += 1
                    self.parameters[mx_name].append(ind)
                mx[ind] = np.nan
            mx[ind[::-1]] = mx[ind]

    def build_beta(self):
        """Sets up a zero Beta matrix using info on classified variables.
        
        Returns:
            
            A zero Beta matrix and a tuple of correspondin rows and columns
            names.
        """
        rows = self.vars['SPart']
        cols = rows
        n = len(rows)
        return np.zeros((n, n)), (rows, cols)

    def build_lambda(self):
        """Sets up a zero Lambda matrix using info on classified variables.

        Returns:
            
        A zero Lambda matrix and a tuple of correspondin rows and columns
        names.
        """
        rows = self.vars['IndsObs']
        cols = self.vars['SPart']
        n = len(rows)
        m = len(cols)
        return np.zeros((n, m)), (rows, cols)

    def build_psi(self):
        """Sets up a zero Psi matrix using info on classified variables.
        
        Returns:
        
            A zero Psi matrix and a tuple of correspondin rows and columns
            names.
        """
        rows = self.vars['SPart']
        cols = rows
        n = len(rows)
        return np.zeros((n, n)), (rows, cols)

    def build_theta(self):
        """Sets up a zero Theta matrix using info on classified variables.
        
        Returns:
        
            A zero Theta matrix and a tuple of correspondin rows and columns
            names.
        """
        rows = self.vars['IndsObs']
        cols = rows
        n = len(rows)
        return np.zeros((n, n)), (rows, cols)

    def prepare_diff_matrices(self):
            """Builds derivatives of each of matricies."""
            self.d_param_matrices = list()
            ms = self.matrices
            symmetrics = self.mx_names_symm
            for mx in self.mx_names:
                for i, j in self.parameters[mx]:
                    dMt = np.zeros_like(ms[mx])
                    dMt[i, j] = 1
                    if mx in symmetrics:
                        dMt[j, i] = 1
                    self.d_param_matrices.append((mx[0], dMt))

    def get_bounds(self):
        mx_to_fix = ['Psi', 'Theta']
        bounds = list()
        for mx in self.mx_names:
            for i, j in self.parameters[mx]:
                if i == j and mx in mx_to_fix:
                    bounds.append((0, None))
                else:
                    bounds.append((None, None))
        return bounds

    def calculate_sigma(self, mx_beta, mx_lambda, mx_psi, mx_theta):
        """Calculates Sigma matrix.
        
        Keyword arguments:
            
            mx_beta   -- Beta matrix.
            
            mx_lambda -- Lambda matrix.
            
            mx_psi    -- Psi matrix.
            
            mx_theta  -- Theta matrix.
            
        Returns:
            
            Sigma matrix and pair (Lambda (E - B)^-1, (E - B)^-1) for auxilary
            purposes.
        """
        c = np.linalg.inv(np.identity(mx_beta.shape[0]) - mx_beta)
        m = mx_lambda @ c
        return m @ mx_psi @ m.T + mx_theta, (m, c)

    def calculate_sigma_gradient(self, mx_beta, mx_lambda, mx_psi, mx_theta,
                                 m=None, c=None):
        """Calculates Sigma matrix gradient.
        
        Keyword arguments:
            
            mx_beta   -- Beta matrix.
            
            mx_lambda -- Lambda matrix.
            
            mx_psi    -- Psi matrix.
            
            mx_theta  -- Theta matrix.
            
            m         -- Lambda (E - B)^-1 if precalculated.
            
            c         -- (E - B)^-1 if precalculated.
            
        Returns:
            
            List of Sigma derivatives w.r.t to parameters in order as specified
            in self.params and self.parameters.
        """
        if c is None:
            c = np.linalg.inv(np.identity(mx_beta.shape[0]) - mx_beta)
            m = mx_lambda @ c
        m_t = m.T
        p = c @ mx_psi
        d = p @ m_t
        grad = list()
        for mxType, mx in self.d_param_matrices:
            if mxType == 'T':
                grad.append(mx)
            elif mxType == 'L':
                t = mx @ d
                grad.append(t + t.T)
            elif mxType == 'B':
                t = mx @ p
                grad.append(m @ (t + t.T) @ m_t)
            elif mxType == 'P':
                grad.append(m @ mx @ m_t)
            else:
                grad.append(np.zeros_like(mx_theta))
        return grad


    def calculate_sigma_hessian(self, mx_beta, mx_lambda, mx_psi, mx_theta, m,
                                c):
        """Calculates Sigma matrix hessian.
        
        Keyword arguments:
            
            mx_beta   -- Beta matrix.
            
            mx_lambda -- Lambda matrix.
            
            mx_psi    -- Psi matrix.
            
            mx_theta  -- Theta matrix.
            
            m         -- Lambda (E - B)^-1 if precalculated.
            
            c         -- (E - B)^-1 if precalculated.
            
        Returns:
            
            Matrix of Sigma 2nd derivatives w.r.t to parameters in order as
            specified in self.params and self.parameters.
        """
        mx_zero = np.zeros_like(mx_theta)
        n, k = self.num_params, mx_theta.shape[0]
        hessian = np.zeros((n, n, k, k))
        m_t = m.T
        c_psi = c @ mx_psi
        c_psi_t = c_psi.T
        t = c_psi @ c.T
        for i, j in np.nditer(np.triu_indices(n)):
            a_type, a_mx = self.d_param_matrices[i]
            b_type, b_mx = self.d_param_matrices[j]
            if (a_type == 'Psi' and (b_type in ('Beta', 'Lambda'))) or\
               (a_type == 'Lambda' and (b_type in ('Beta'))):
                a_type, b_type, a_mx, b_mx = b_type, a_type, b_mx, a_mx
            if a_type == 'Beta':
                if b_type == 'Beta':
                    k = a_mx @ c_psi
                    k_sum = k + k.T
                    bac = a_mx @ c
                    bac_t = bac.T
                    bbc = b_mx @ c
                    bbc_t = bbc.T
                    h = m @ (bbc @ k_sum + k_sum @ bbc_t + bac @ bbc @ mx_psi +
                             c_psi_t @ bbc_t @ bac_t) @ m_t
                    hessian[i, j] = h
                elif b_type == 'Lambda':
                    k = a_mx @ c_psi
                    k_sum = k + k.T
                    t = b_mx @ c
                    hessian[i, j] = m @ k_sum @ t.T + t @ k_sum @ m_t
                elif b_type == 'Psi':
                    k_hat = a_mx @ c @ b_mx
                    hessian[i, j] = m @ (k_hat + k_hat.T) @ m_t
                elif b_type == 'Theta':
                    hessian[i, j] = mx_zero
            elif a_type == 'Lambda':
                if b_type == 'Lambda':
                    hessian[i, j] = a_mx @ t @ b_mx.T + b_mx @ t @ a_mx.T
                elif b_type == 'Psi':
                    ma = a_mx @ c
                    hessian[i, j] = ma @ b_mx @ m_t + m @ b_mx @ ma.T
                elif b_type == 'Theta':
                    hessian[i, j] = mx_zero
            elif a_type == 'Psi':
                hessian[i, j] = mx_zero
            else:
                hessian[i, j] = mx_zero
        hessian += np.triu(hessian, 1)
        return hessian
