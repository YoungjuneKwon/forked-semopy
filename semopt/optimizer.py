from .Portmin.optimization import minimize as portmin
from .utils import chol_inv
from scipy.optimize import minimize
from pandas import DataFrame
from .model import Model
import numpy as np


class Optimizer:
    def __init__(self, mod: Model):
        """Creates a Optimiser instance.

        Keywrod arguments:
        mod -- A SEM Model.
        """
        self.bounds = mod.get_bounds()
        self.model = mod
        if mod.param_vals is not None:
            self.load_empirical_data()
        else:
            self.params = None

    def optimize(self, objective='MLW', regu=None, a=1.0, params_to_pen=list(),
                 method='SLSQP', **args):
        """Optimize model.

        Keyword arguments:
        objective     -- Name of objective function to minimize, i.e. "MLW",
                        "ULS", "GLS", et cetera.
        regu          -- A name of regularization to apply ("l1", "l2", None).
        a             -- A regularization multiplier.
        params_to_pen -- A list of indices of params to be penalized.
        method        -- Name of an optimization technique to apply, i.e.
                         "SLSQP", "Adam", "SMSNO".

        Returns:
        Objective function value and regularization function value (if regu
        is not None).
        """
        scipy_methods = {'SLSQP'}
        stochastic_methods = {'Adam': self.minimize_adam,
                              'Momentum': self.minimize_momentum,
                              'Nesterov': self.minimize_nesterov,
                              'SGD': self.minimize_sgd}
        options = {'maxiter': 1e3, 'ftol': 1e-8}
        if self.params is None:
            print("Error: dataset must be provided before optimisation.")
            return None
        lf, of, rf, grad = self.compose_loss_function(objective, regu,
                                                      params_to_pen, a)
        portmin_methods = {'SMSNO': {},
                           'SUMSL': {'grad': grad},
                           'HUSML': {'grad': grad, 'hess': None}}
        if method in scipy_methods:
            res = minimize(lf, self.params, options=options, jac=grad,
                           bounds=self.bounds, method=method, **args)
            self.params = res.x
        elif method in stochastic_methods:
            self.params = stochastic_methods[method](lf, grad, self.params,
                                                     **args)
        elif method in portmin_methods:
            self.params = portmin(lf, self.params, **portmin_methods[method])
        else:
            raise Exception("Unkown optimisation method {}.".format(method))
        return lf(self.params) if rf is None else (lf(self.params),
                                                   rf(self.params))

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
        f = mod.calculate_sigma_gradient
        self.get_sigma_grad = lambda m, c: f(self.mx_beta, self.mx_lambda,
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
        if name in gradDict:
            return gradDict[name]
        else:
            return None

    def compose_gradient_function(self, method: str, regu=None, a=1.0,
                                  paramsToPen=None):
        """ Builds a gradient function if possible. """
        grad = None
        grad_of = self.get_gradient_function(method)
        if grad_of is not None:
            grad = grad_of
        if regu is not None and grad is not None:
            regu = self.get_gradient_function(regu)
            if regu is None:
                return None

            def grad_composed(params):
                g = grad_of(params)
                rg = a * regu(params)
                mask = np.ones(len(params), np.bool)
                mask[paramsToPen] = False
                rg[mask] = 0
                return g + rg
            grad = grad_composed
        return grad

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
        (Loss function, obj_func, a * regularization)
        Loss function = obj_func + a * regularization"""
        obj_func = self.get_obj_function(method)
        regu = None
        if regularization is not None:
            regu = self.get_regularization(regularization, params_to_pen)
            loss_func = lambda params: obj_func(params) + a * regu(params)
        else:
            loss_func = obj_func
        grad = self.compose_gradient_function(method, regularization, a,
                                              params_to_pen)
        return (loss_func, obj_func, regu, grad)

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
                      chunk_size=75, num_epochs=60, memory=True):
        cov = self.mx_cov.copy()

        def iteration(j, m_t, v_t, b1_t, b2_t, x):
            self.mx_cov = np.cov(self.profiles[j:j+chunk_size, :],
                                 rowvar=False, bias=True)
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
        for i in range(num_epochs):
            np.random.shuffle(self.profiles)
            for j in range(0, len(self.profiles), chunk_size):
                ret = iteration(j, m_t, v_t, b1_t, b2_t, x)
                if ret is not None:
                    m_t, v_t, b1_t, b2_t, x, f_val = ret
                    if memory and f_val < best_f_val:
                        best_f_val = f_val
                        best_x = x.copy()
        return best_x if memory else x

    def minimize_momentum(self, lf, grad, x0, step=0.002, resistance=0.9,
                          num_epochs=100, chunk_size=25, memory=True):
        cov = self.mx_cov.copy()

        def iteration(j,  x, v_t):
            self.mx_cov = np.cov(self.profiles[j:j+chunk_size, :],
                                 rowvar=False, bias=True)
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
        for i in range(num_epochs):
            np.random.shuffle(self.profiles)
            for j in range(0, len(self.profiles), chunk_size):
                it = iteration(j, x, v_t)
                if it is not None:
                    x, v_t, f_val = it
                    if memory and f_val < best_f_val:
                        best_f_val = f_val
                        best_x = x.copy()
        return best_x if memory else x

    def minimize_sgd(self, lf, grad, x0, step=0.5, num_epochs=100,
                     chunk_size=50, memory=True):
        cov = self.mx_cov.copy()
        x = x0.copy()
        best_f_val = lf(x0)
        best_x = x0
        for i in range(num_epochs):
            np.random.shuffle(self.profiles)
            for j in range(0, len(self.profiles), chunk_size):
                self.mx_cov = np.cov(self.profiles[j:j+chunk_size, :],
                                     rowvar=False, bias=True)
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
                          num_epochs=30, chunk_size=75, memory=True):
        cov = self.mx_cov.copy()

        def iteration(j, x, v_t):
            self.mx_cov = np.cov(self.profiles[j:j+chunk_size, :],
                                 rowvar=False, bias=True)
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
        for i in range(num_epochs):
            np.random.shuffle(self.profiles)
            for j in range(0, len(self.profiles), chunk_size):
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
            return np.nan
        s, logdet_sigma = np.linalg.slogdet(sigma)
        if s < 0:
            return np.nan
        log_det_ratio = logdet_sigma - self.cov_logdet
        tr = np.einsum('ij,ji->', self.mx_cov, inv_sigma) - sigma.shape[0]
        loss = tr + log_det_ratio
        if loss < 0 or loss > 1e3:
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
        n = len(params)
        Sigma = self.calculate_sigma(params)
        Sigma_grad = self.calculate_sigma_gradient(params)
        Sigma_hess = self.calculate_sigma_hessian(params)
        Cov = self.m_cov
        inv_Sigma = np.linalg.inv(Sigma)
        A = inv_Sigma @ Cov
        B = np.identity(Cov.shape[0]) - A
        hessian = np.zeros((n, n))
        for i in range(n):
            C = inv_Sigma @ Sigma_grad[i]
            for k in range(i, n):
                t = Sigma_grad[k] @ C
                hessian[i, k] = np.trace(inv_Sigma @ ((Sigma_hess[i, k] - t) @ B + t.T @ A))
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
        loss = np.einsum('ij,ji->', t, t)
        return loss

    def general_least_squares(self, params):
        self.update_matrices(params)
        try:
            sigma, _ = self.get_sigma()
        except np.linalg.LinAlgError:
            return np.nan
        t = (sigma - self.mx_cov) @ self.mx_cov_inv
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
        return np.array([2 * np.einsum('ij,ji->', g, t)
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
        t = self.mx_cov_inv @ (sigma - self.mx_cov) @ self.mx_cov_inv
        return np.array([2 * np.einsum('ij,ji->', g, t)
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
