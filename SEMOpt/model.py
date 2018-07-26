from itertools import combinations, product, chain
from .model_abstract import ModelABC
from scipy.stats import linregress
from pandas import DataFrame
import numpy as np


class Model(ModelABC):
    def __init__(self, model_description: str, description=None,
                 force_load=set()):
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
        ops = Model.operations
        for v in chain(self.vars['ObsEndo'], self.vars['Latents'],
                       self.vars['Indicators']):
            if v not in description[v][ops.COVARIANCE]:
                description[v][ops.COVARIANCE][v] = list()
        for v1, v2 in chain(combinations(self.vars['LatExo'], 2),
                            combinations(self.vars['Outputs'], 2)):
            if v1 not in description[v2][ops.COVARIANCE] and\
               v2 not in description[v1][ops.COVARIANCE]:
                   description[v1][ops.COVARIANCE][v2] = list()
        super().prepare_parameters(description)
        for mx in self.parameters:
            self.parameters[mx] = sorted(self.parameters[mx])
        for v in self.vars['Observed']:
            i, j = self.lambda_names[0].index(v), self.lambda_names[1].index(v)
            self.mx_lambda[i, j] = 1.0

    def load_dataset(self, data: DataFrame, bias=True, center=True):
        """Loads dataset and applies starting values.

        Keyword arguments:
        data   -- A Pandas' DataFrame containing data.
        bias   -- Shall we calculated covariance matrix using np.cov with
                  bias=True or bias=False.
        center -- Shall we center data (substract means along cols)?
        """
        super().load_dataset(data, bias, center)
        cov = DataFrame(self.mx_cov, index=self.vars['IndsObs'],
                        columns=self.vars['IndsObs'])
        for v1, v2 in chain(product(self.vars['ObsExo'], repeat=2),
                            self.fixed_covars):
            ind = (self.psi_names[0].index(v1), self.psi_names[1].index(v2))
            self.mx_psi[ind] = cov[v1][v2]
            self.mx_psi[ind[::-1]] = self.mx_psi[ind]
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
#                l, r = self.beta_names[0][ind[0]], self.beta_names[1][ind[1]]
#                ml, mr = 1.0, 1.0
#                if l in self.vars['Latents']:
#                    i = self.lambda_names[1].index(l)
#                    l = self.first_indicators[l]
#                    j = self.lambda_names[0].index(l)
#                    ml = self.mx_lambda[j, i]
#                if r in self.vars['Latents']:
#                    i = self.lambda_names[1].index(r)
#                    r = self.first_indicators[r]
#                    j = self.lambda_names[0].index(r)
#                    mr = self.mx_lambda[j, i]
#                slope = linregress(mr * data[r], ml * data[l]).slope
                self.mx_beta[ind] = 0#slope
            self.param_vals[k] = self.mx_beta[ind]
            k += 1
        self.beta_range = (self.beta_range, k)
        self.lambda_range = k
        for ind in self.parameters['Lambda']:
            params_inds = np.append(params_inds, ind)
            if np.isnan(self.mx_lambda[ind]):
                l, r = self.lambda_names[0][ind[0]], self.lambda_names[1][ind[1]]
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
                    ## TODO: check if cov instead of zero in case of observable
                    ## variables works better.
            self.mx_psi[ind[::-1]] = self.mx_psi[ind]
            self.param_vals[k] = self.mx_psi[ind]
            k += 1
        self.psi_range = (self.psi_range, k)
        self.theta_range = k
        for ind in self.parameters['Theta']:
            l, r = self.theta_names[0][ind[0]], self.theta_names[1][ind[1]]
            params_inds = np.append(params_inds, ind)
            self.mx_theta[ind] = cov[l][r] / 2
            self.param_vals[k] = self.mx_theta[ind]
            k += 1
        self.theta_range = (self.theta_range, k)
        params_inds = np.reshape(params_inds, (k, 2))
        self.beta_params_inds = params_inds[self.beta_range[0]:self.beta_range[1], :]
        self.lambda_params_inds = params_inds[self.lambda_range[0]:self.lambda_range[1], :]
        self.psi_params_inds = params_inds[self.psi_range[0]:self.psi_range[1], :]
        self.psi_params_inds_t = self.psi_params_inds[:, ::-1]
        self.theta_params_inds = params_inds[self.theta_range[0]:self.theta_range[1], :]

    def parse_operation(self, op, lvalue, rvalue, args):
        ops = Model.operations
        if op == ops.REGRESSION:
            ind_lv = self.beta_names[0].index(lvalue)
            ind_rv = self.beta_names[1].index(rvalue)
            ind = (ind_lv, ind_rv)
            if ind in self.parameters['Beta']:
                raise Exception("Operation {} {} {} already specified.".format(lvalue, op, rvalue))
            try:
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
                mx[ind] = float(args[0])
            except Exception:
                if len(args) and args[0] == 'sv':
                    self.fixed_covars.append((lvalue, rvalue))
                else:
                    self.num_params += 1
                    self.parameters[mx_name].append(ind)
                    mx[ind] = np.nan
        else:
            raise NotImplementedError("{} is an uknown opcode.".format(op))

    def build_beta(self):
        """
        Sets up a zero Beta matrix using info on classified variables.

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
                    bounds.append((None, None))
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
        n, m = self.num_params, mx_theta.shape[0]
        hessian = np.zeros((n, n, m, m))
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
        hessian += np.triu(hessian, 1).T
        return hessian
