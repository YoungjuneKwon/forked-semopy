import political_democracy as pd
import unittest
from pandas import DataFrame
import numpy as np
import optimizer
import model

class test_opt_regression_case(unittest.TestCase):
    def setUp(self):
        mod = 'y ~ x'
        self.coeff = 3
        x = [10 + np.random.rand() for _ in range(40)]
        y = [self.coeff * x for x in x]
        d = DataFrame(np.array([x, y]).T, columns=['x', 'y'])
        self.model = model.Model(mod)
        self.model.load_dataset(d)
        self.opt = optimizer.Optimizer(self.model)

    def are_params_correct(self):
        return abs(self.opt.params[0] - self.coeff) < 0.1

    def test_mlw(self):
        lf_before = self.opt.ml_wishart(self.opt.params)
        lf = self.opt.optimize(objective = 'MLW')
        self.assertTrue(lf < lf_before)
        self.assertTrue(self.are_params_correct())

    def test_uls(self):
        lf_before = self.opt.unweighted_least_squares(self.opt.params)
        lf = self.opt.optimize(objective='ULS')
        self.assertTrue(lf < lf_before)
        self.assertTrue(self.are_params_correct())

    def test_gls(self):
        lf_before = self.opt.general_least_squares(self.opt.params)
        lf = self.opt.optimize(objective='GLS')
        self.assertTrue(lf < lf_before)
        self.assertTrue(self.are_params_correct())     

    def test_size(self):
        # TODO: Probaby remove because model is a subject to change.
        n, m = self.model.mx_beta.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)
        n, m = self.model.mx_psi.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)
        n, m = self.model.mx_lambda.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)

class test_opt_pd_case(unittest.TestCase):

    def setUp(self):
        mod = pd.get_model()
        self.model = model.Model(mod)
        self.data = pd.get_data()
        self.model.load_dataset(self.data)

    def are_params_correct(self):
        # Correct parameter values are not known, thereof we just assume
        # LAVAAN to be somewhat correct.
        pass

    def test_size(self):
        # TODO: Probaby remove because model is a subject to change.
        n, m = self.model.mx_beta.shape
        self.assertEqual(n, 3)
        self.assertEqual(m, 3)
        n, m = self.model.mx_psi.shape
        self.assertEqual(n, 3)
        self.assertEqual(m, 3)
        n, m = self.model.mx_lambda.shape
        self.assertEqual(n, 3)
        self.assertEqual(m, 8)

    def test_variables(self):
        self.assertEquals(len(self.model.vars['Latents']), 3)
        self.assertEquals(len(self.model.vars['All']), 11)
        self.assertTrue('ind60' in self.model.vars['LatExo'])
        self.assertTrue('dem65' in self.model.vars['LatEndo'])
        self.assertTrue('dem60' in self.model.vars['LatEndo'])
        self.assertTrue('dem65' in self.model.vars['Outputs'])

    def test_mlw(self):
        lf_before = self.opt.ml_wishart(self.opt.params)
        lf = self.opt.optimize(objective = 'MLW')
        self.assertTrue(lf < lf_before)
        self.assertTrue(self.are_params_correct())

    def test_uls(self):
        lf_before = self.opt.unweighted_least_squares(self.opt.params)
        lf = self.opt.optimize(objective='ULS')
        self.assertTrue(lf < lf_before)
        self.assertTrue(self.are_params_correct())

    def test_gls(self):
        lf_before = self.opt.general_least_squares(self.opt.params)
        lf = self.opt.optimize(objective='GLS')
        self.assertTrue(lf < lf_before)
        self.assertTrue(self.are_params_correct())

    def test_dataset(self):
        cov = self.model.mx_cov
        m, n = cov.shape
        self.assertEquals(m, 8)
        self.assertEquals(n, 8)
        for i, j in np.nditer(np.triu_indices(m)):
            if i == j:
                self.assertTrue(cov[i, i] > 0)
            else:
                self.assertEquals(cov[i, j], cov[j, i])
        self.assertTrue(np.all(np.linalg.eigvals(cov) > 0))