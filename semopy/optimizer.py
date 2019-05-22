'''Optimizer is utilised for fitting models' parameters to a given objective function.'''
from portmin import minimize as portmin
from .utils import chol_inv
from scipy.optimize import minimize
from pandas import DataFrame
from .model import Model
from .utils import cov as cov_f
from functools import partial
import numpy as np


class Optimizer:
    '''Optimizer class is responsible for estimating a given model's parameters.
    
    Keywrod arguments:
        
            mod -- A SEM Model.
    '''
    def __init__(self, mod: Model):
        """Creates a Optimiser instance.
        
        Keywrod arguments:
            
            mod -- A SEM Model.
        """
        self.omit_warnings = True
        self.bounds = mod.get_bounds()
        self.model = mod
        if mod.param_vals is not None:
            self.load_empirical_data()
        else:
            self.params = None

    def optimize(self, objective='MLW', regu=None, a=1.0, params_to_pen=list(),
                 method='SLSQP', bounds=True, **args):
        """Optimize model.
        
        Keyword arguments:
            
            objective       -- Name of objective function to minimize, i.e.
                               "MLW", "ULS", "GLS", et cetera.
                               
            regu            -- A name of regularization to apply
                               ("l1", "l2", None).
                               
            a               -- A regularization multiplier.
            
            params_to_pen   -- A list of indices of params to be penalized.
            
            method          -- Name of an optimization technique to apply, i.e.
                               "SLSQP", "Adam", "SMSNO".
                               
            bounds          -- Whether to use bound constraints if an optimizer
                               supports it. "True" if use default model bounds,
                               "False" or "None" for no bounds, list of
                               (a, b) tuples for custom bounds.
                               
        Returns:
            
            Objective function value and regularization function value (if regu
            is not None).
        """
        scipy_methods = {'SLSQP', 'L-BFGS-B'}
        stochastic_methods = {'Adam': self.minimize_adam,
                              'Momentum': self.minimize_momentum,
                              'Nesterov': self.minimize_nesterov,
                              'SGD': self.minimize_sgd}
        options = {'maxiter': 1e3, 'ftol': 1e-8}
        if self.params is None:
            print("Error: dataset must be provided before optimisation.")
            return None

        lf, of, rf, grad, hess = self.compose_loss_function(objective, regu,
                                                            params_to_pen, a)
        portmin_methods = {'SMSNO': {},
                           'SUMSL': {'grad': grad},
                           'HUSML': {'grad': grad, 'hess': hess}}
        if bounds is True:
            bounds = self.bounds
        elif type(bounds) == list:
            pass
        else:
            bounds = None
        if bounds:
            for i, (a, b) in enumerate(bounds):
                if a and self.params[i] < a:
                    self.params[i] = a + 1e-3
                    if b and self.params[i] > b:
                        self.params[i] = b
                elif b and self.params[i] > b:
                    self.params[i] = b - 1e-3
                    if a and self.params[i] > a:
                        self.params[i] = a
        if method in scipy_methods:
            res = minimize(lf, self.params, options=options, jac=grad,
                           bounds=bounds, method=method, **args)
            self.params = res.x
        elif method in stochastic_methods:
            self.params = stochastic_methods[method](lf, grad, self.params,
                                                     **args)
        elif method in portmin_methods:
           self.params = portmin(lf, self.params, **portmin_methods[method])
#            raise Exception("PORTMIN functionality temporarily removed.")
        else:
            raise Exception("Unkown optimisation method {}.".format(method))
        lf_t = lf(self.params)
        self.last_run = (objective, regu, a, params_to_pen, method), lf_t
        return lf_t if rf is None else (lf_t, rf(self.params))

    def apply_degeneracy(self, degenerate_vars: list):
        """Removes degenerate_vars from final Sigma computations in both
        get_sigma() and get_sigma_grad().
        
        Keyword arguments:
            
            degenerate_vars -- A list of observed variables to exclude from
                               both empirical and model-induced covariance
                               matrices.
                               
        """
        try:
            self.mx_cov = self._mx_cov
            self.mx_cov_inv = self._mx_cov_inv
            self.mx_covlike_identity = self._mx_covlike_identity
            self.get_sigma = self._get_sigma
            self.get_sigma_grad = self._get_sigma_grad
            self.cov_logdet = self._cov_logdet
        except Exception:
            pass
        if degenerate_vars:
            obs = self.model.vars['IndsObs']
            exclude = [obs.index(v) for v in degenerate_vars]
            self._get_sigma = self.get_sigma
            self._get_sigma_grad = self.get_sigma_grad
            self._get_sigma_hess = self.get_sigma_hess
            self.get_sigma = partial(self.get_degenerate_sigma,
                                     exclude=exclude)
            self.get_sigma_grad = partial(self.get_degenerate_sigma_grad,
                                          exclude=exclude)
            self._mx_cov = self.mx_cov
            self.mx_cov = self.turn_mx_degenerate(self.mx_cov, exclude)
            self._mx_cov_inv = self.mx_cov_inv
            self.mx_cov_inv = np.linalg.inv(self.mx_cov)
            self._mx_covlike_identity = self.mx_covlike_identity
            self.mx_covlike_identity = np.identity(self.mx_cov.shape[0])
            self._cov_logdet = self.cov_logdet
            self.cov_logdet = np.linalg.slogdet(self.mx_cov)[1]
                       

    def load_dataset(self, data: DataFrame):
        """Loads a dataset in case if Model.load_dataset hasn't been called
           yet.
           
        Keyword arguments:
            
            data -- A pandas' DataFrame.
        """
        self.model.load_dataset(data)
        self.load_empirical_data()

    def load_empirical_data(self):
        mod = self.model
        self.mx_cov = mod.mx_cov
        self.mx_cov_inv = np.linalg.inv(self.mx_cov)
        self.mx_covlike_identity = np.identity(self.mx_cov.shape[0])
        self.profiles = mod.raw_data
        self.cov_logdet = np.linalg.slogdet(self.mx_cov)[1]
        self.starting_params = mod.param_vals.copy()
        self.params = self.starting_params.copy()
        self.mx_beta = mod.matrices['Beta'].copy()
        self.mx_lambda = mod.matrices['Lambda'].copy()
        self.mx_psi = mod.matrices['Psi'].copy()
        self.mx_theta = mod.matrices['Theta'].copy()
        self.update_matrices = lambda params:\
                               mod.apply_parameters(params,
                                                    self.mx_beta,
                                                    self.mx_lambda,
                                                    self.mx_psi, self.mx_theta)
        self.get_sigma = lambda: mod.calculate_sigma(self.mx_beta,
                                                     self.mx_lambda,
                                                     self.mx_psi,
                                                     self.mx_theta)
        f1 = mod.calculate_sigma_gradient
        self.get_sigma_grad = lambda m, c: f1(self.mx_beta, self.mx_lambda,
                                              self.mx_psi, self.mx_theta,
                                              m, c)
        f2 = mod.calculate_sigma_hessian
        self.get_sigma_hess = lambda m, c: f2(self.mx_beta, self.mx_lambda,
                                              self.mx_psi, self.mx_theta,
                                              m, c)

    def get_obj_function(self,  name: str):
        objDict = {'ULS': self.unweighted_least_squares,
                   'GLS': self.general_least_squares,
                   'MLW': self.ml_wishart}
        try:
            return objDict[name]
        except KeyError:
            raise Exception("Unknown optimization method {}.".format(name))

    def get_gradient_function(self, name: str):
        gradDict = {'MLW': self.ml_wishart_gradient,
                    'ULS': self.uls_gradient,
                    'GLS': self.gls_gradient,
                    'l1':  self.regu_l1_gradient,
                    'l2':  self.regu_l2_gradient}
        return gradDict.get(name, None)

    def get_hessian_function(self, name: str):
        hessDict = {'MLW': self.ml_wishart_hessian,
                    'l2':  self.regu_l2_hessian}
        return hessDict.get(name, None)

    def compose_diff_function(self, method: str, hess=False, regu=None, a=1.0,
                              paramsToPen=None):
        """ Builds a gradient or a hessian function if possible. """
        if hess:
            getter = self.get_hessian_function
        else:
            getter = self.get_gradient_function
        df = None
        df_of = getter(method)
        if df_of is not None:
            df = df_of
        if regu is not None and df is not None:
            regu = getter(regu)
            if regu is None:
                return None

            def df_composed(params):
                g = df_of(params)
                rg = a * regu(params)
                mask = np.ones(len(params), np.bool)
                mask[paramsToPen] = False
                rg[mask] = 0
                return g + rg
            df = df_composed
        return df

    def compose_loss_function(self, method: str, regularization=None,
                              params_to_pen=None, a=1.0):
        """Build a loss function.
        
        Key arguments:
            
            method         -- a name of an optimization technique to apply.
            
            regularization -- a name of regularizatio technique to apply.
            
            params_to_pen  -- indicies of parameters from params' vector to
                              penalize. If None, then all params are penalized.
                              
            a              -- a regularization multiplier.
            
        Returns:
            
            (Loss function, obj_func, a * regularization, gradient of loss
            function, hessian of loss function)
            Loss function = obj_func + a * regularization
        """
        obj_func = self.get_obj_function(method)
        regu = None
        if regularization is not None:
            regu = self.get_regularization(regularization, params_to_pen)
            loss_func = lambda params: obj_func(params) + a * regu(params)
        else:
            loss_func = obj_func
        grad = self.compose_diff_function(method, False, regularization, a,
                                          params_to_pen)
        hess = self.compose_diff_function(method, True, regularization, a,
                                          params_to_pen)
        return (loss_func, obj_func, regu, grad, hess)

    def get_regularization(self, name: str, params_to_pen: list):
        reguDict = {'l1': self.regu_l1,
                    'l2': self.regu_l2}
        try:
            f = reguDict[name]
            if params_to_pen is None or len(params_to_pen) == 0:
                return f
            else:
                def regu_wrapper(params):
                    params = params[regu_wrapper.params_to_pen]
                    return regu_wrapper.f(params)
                regu_wrapper.f = reguDict[name]
                regu_wrapper.params_to_pen = params_to_pen
                return regu_wrapper
        except KeyError:
            raise Exception("Unknown regularization method {}.".format(name))

    def minimize_adam(self, lf, grad, x0, step=1.0, beta1=0.9, beta2=0.999,
                      chunk_size=None, num_epochs=1000, memory=True):
        cov = self.mx_cov.copy()
            
        def iteration(j, m_t, v_t, b1_t, b2_t, x):
            nonlocal chunk_size
            if chunk_size:
                self.mx_cov = cov_f(self.profiles[j:j+chunk_size, :])
            g = grad(x)
            m_t = beta1 * m_t + (1 - beta1) * g
            v_t = beta2 * v_t + (1 - beta1) * (g ** 2)
            b1_t *= beta1
            b2_t *= beta2
            m_t_hat, v_t_hat = m_t / (1 - b1_t), v_t / (1 - b2_t)
            x = x - step / (np.sqrt(v_t_hat) + 1e-8) * m_t_hat
            self.mx_cov = cov
            f_val = lf(x)
            if np.isnan(f_val):
                return None
            return m_t, v_t, b1_t, b2_t, x, f_val

        x = x0.copy()
        n = len(x)
        m_t, v_t, b1_t, b2_t = np.zeros(n), np.zeros(n), 1, 1
        best_f_val = lf(x)
        best_x = x.copy()
        step_size = chunk_size if chunk_size else len(self.profiles)
        for i in range(num_epochs):
            if chunk_size:
                np.random.shuffle(self.profiles)
            for j in range(0, len(self.profiles), step_size):
                ret = iteration(j, m_t, v_t, b1_t, b2_t, x)
                if ret is not None:
                    m_t, v_t, b1_t, b2_t, x, f_val = ret
                    if memory and f_val < best_f_val:
                        best_f_val = f_val
                        best_x = x.copy()
        return best_x if memory else x

    def minimize_momentum(self, lf, grad, x0, step=0.002, resistance=0.9,
                          num_epochs=200, chunk_size=None, memory=True):
        cov = self.mx_cov.copy()

        def iteration(j,  x, v_t):
            nonlocal chunk_size
            if chunk_size:
                self.mx_cov = cov_f(self.profiles[j:j+chunk_size, :])
            v_tn = resistance * v_t + step * grad(x)
            x_next = x - v_tn
            self.mx_cov = cov
            f_val = lf(x_next)
            if np.isnan(f_val):
                return None
            return x_next, v_tn, f_val
        x = x0.copy()
        v_t = np.zeros(len(x))
        best_f_val = lf(x)
        best_x = x.copy()
        step_size = chunk_size if chunk_size else len(self.profiles)
        for i in range(num_epochs):
            if chunk_size:
                np.random.shuffle(self.profiles)
            for j in range(0, len(self.profiles), step_size):
                it = iteration(j, x, v_t)
                if it is not None:
                    x, v_t, f_val = it
                    if memory and f_val < best_f_val:
                        best_f_val = f_val
                        best_x = x.copy()
        return best_x if memory else x

    def minimize_sgd(self, lf, grad, x0, step=0.5, num_epochs=100,
                     chunk_size=None, memory=True):
        cov = self.mx_cov.copy()
        x = x0.copy()
        best_f_val = lf(x0)
        best_x = x0
        step_size = chunk_size if chunk_size else len(self.profiles)
        for i in range(num_epochs):
            if chunk_size:
                np.random.shuffle(self.profiles)
            for j in range(0, len(self.profiles), step_size):
                if chunk_size:
                    self.mx_cov = cov_f(self.profiles[j:j+chunk_size, :])
                tx = x - step * grad(x)
                self.mx_cov = cov
                f_val = lf(tx)
                if not np.isnan(f_val):
                    x = tx
                    if memory and f_val < best_f_val:
                        best_f_val = f_val
                        best_x = x.copy()
        return best_x if memory else x

    def minimize_nesterov(self, lf, grad, x0, step=0.002, resistance=0.9,
                          num_epochs=200, chunk_size=None, memory=True):
        cov = self.mx_cov.copy()

        def iteration(j, x, v_t):
            nonlocal chunk_size
            if chunk_size:
                self.mx_cov = cov_f(self.profiles[j:j+chunk_size, :])
            try:
                v_tn = resistance * v_t + step * grad(x - resistance * v_t)
            except np.linalg.LinAlgError:
                self.mx_cov = cov
                return None
            x_next = x - v_tn
            self.mx_cov = cov
            f_val = lf(x_next)
            if np.isnan(f_val):
                return None
            return x_next, v_tn, f_val
        x = x0.copy()
        v_t = np.zeros(len(x))
        best_f_val = lf(x)
        best_x = x.copy()
        step_size = chunk_size if chunk_size else len(self.profiles)
        for i in range(num_epochs):
            if chunk_size:
                np.random.shuffle(self.profiles)
            for j in range(0, len(self.profiles), step_size):
                it = iteration(j, x, v_t)
                if it is not None:
                    x, v_t, f_val = it
                    if memory and f_val < best_f_val:
                        best_f_val = f_val
                        best_x = x.copy()
        return best_x if memory else x

    def ml_wishart(self, params):
        """Computes wishart likelihood ratio.
        
        Keyword arguments:
            
            params -- a parameters values' vector.
            
        Returns:
            
            MLR, nan if failed to compute at (.)params.
        """
        self.update_matrices(params)
        try:
            sigma, _ = self.get_sigma()
            inv_sigma = chol_inv(sigma)
        except np.linalg.LinAlgError:
            if not self.omit_warnings:
                print("Warning: MLR(Wishart) -- couldn't compute inverse for Sigma")
            return np.nan
        s, logdet_sigma = np.linalg.slogdet(sigma)
        if s < 0:
            if not self.omit_warnings:
                print("Warning: MLR(Wishart) -- determinant of Sigma is negative.")
            return np.nan
        log_det_ratio = logdet_sigma - self.cov_logdet
        tr = np.einsum('ij,ji->', self.mx_cov, inv_sigma) - sigma.shape[0]
        loss = tr + log_det_ratio
        # Realistically should never happen.
        if loss < 0:
            if not self.omit_warnings:
                print("Warning: MLR(Wishart) -- Negative value.")
            return np.nan
        return loss

    def ml_wishart_gradient(self, params):
        self.update_matrices(params)
        try:
            sigma, (m, c) = self.get_sigma()
            inv_sigma = chol_inv(sigma)
        except np.linalg.LinAlgError:
            t = np.zeros((len(params),))
            t[:] = np.nan
            return t
        sigma_grad = self.get_sigma_grad(m, c)
        cs = inv_sigma - inv_sigma @ self.mx_cov @ inv_sigma
        return np.array([np.einsum('ij,ji->', cs, g)
                         for g in sigma_grad])

    def ml_wishart_hessian(self, params, fortranize=True):
        self.update_matrices(params)
        n = len(params)
        sigma, (m, c) = self.get_sigma()
        sigma_grad = self.get_sigma_grad(m, c)
        sigma_hess = self.get_sigma_hess(m, c)
        cov = self.mx_cov
        inv_sigma = chol_inv(sigma)
        A = inv_sigma @ cov
        B = self.mx_covlike_identity - A
        hessian = np.zeros((n, n))
        for i in range(n):
            C = inv_sigma @ sigma_grad[i]
            for k in range(i, n):
                t = sigma_grad[k] @ C
                hessian[i, k] = np.trace(inv_sigma @ ((sigma_hess[i, k] - t) @ B + t.T @ A))
        hessian += np.triu(hessian, 1).T
        if fortranize:
            return hessian[np.tril_indices(n)]
        else:
            return hessian

    def unweighted_least_squares(self, params):
        self.update_matrices(params)
        try:
            sigma, _ = self.get_sigma()
        except np.linalg.LinAlgError:
            return np.nan
        t = sigma - self.mx_cov
        loss = np.einsum('ij,ij->', t, t)
        return loss

    def general_least_squares(self, params):
        self.update_matrices(params)
        try:
            sigma, _ = self.get_sigma()
        except np.linalg.LinAlgError:
            return np.nan
        t = sigma @ self.mx_cov_inv - self.mx_covlike_identity
        return np.einsum('ij,ji->', t, t)

    def uls_gradient(self, params):
        self.update_matrices(params)
        try:
            sigma, (m, c) = self.get_sigma()
        except np.linalg.LinAlgError:
            t = np.zeros((len(params),))
            t[:] = np.nan
            return t
        sigma_grad = self.get_sigma_grad(m, c)
        t = sigma - self.mx_cov
        return 2 * np.array([np.einsum('ij,ji->', g, t)
                             for g in sigma_grad])

    def gls_gradient(self, params):
        self.update_matrices(params)
        try:
            sigma, (m, c) = self.get_sigma()
        except np.linalg.LinAlgError:
            t = np.zeros((len(params),))
            t[:] = np.nan
            return t
        sigma_grad = self.get_sigma_grad(m, c)
        t = self.mx_cov_inv @ \
            (sigma @ self.mx_cov_inv - self.mx_covlike_identity)
        return 2 * np.array([np.einsum('ij,ji->', g, t)
                             for g in sigma_grad])

    def regu_l1(self, params):
        return np.mean((np.abs(params)))

    def regu_l1_gradient(self, params):
        return np.sign(params) / len(params)

    def regu_l2(self, params):
        return np.linalg.norm(params) ** 2

    def regu_l2_gradient(self, params):
        return 2 * params

    def regu_l2_hessian(self, params):
        return 2 * np.identity(len(params))

    def turn_mx_degenerate(self, mx, exclude):
        return np.delete(np.delete(mx, exclude, axis=0), exclude, axis=1)

    def get_degenerate_sigma(self, exclude: list):
        sigma, (m, c) = self._get_sigma()
        sigma = self.turn_mx_degenerate(sigma, exclude)
        return sigma, (m, c)

    def get_degenerate_sigma_grad(self, m, c, exclude: list):
        sigma_grad = self._get_sigma_grad(m, c)
        return [self.turn_mx_degenerate(g, exclude)
                for g in sigma_grad]
        