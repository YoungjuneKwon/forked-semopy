import unittest
import political_democracy as pd
from pandas import DataFrame
import numpy as np
import model

class test_model_regression_case(unittest.TestCase):

    def setUp(self):
        mod = 'y ~ x'
        self.model = model.Model(mod)

    def test_size(self):
        n, m = self.model.mx_beta.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)
        n, m = self.model.mx_psi.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)
        n, m = self.model.mx_lambda.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)

    def test_variables(self):
        self.assertFalse(self.model.vars['Latents'])
        self.assertEquals(len(self.model.vars['All']), 2)
        self.assertTrue('y' in self.model.vars['ObsEndo'])
        self.assertTrue('x' in self.model.vars['ObsExo'])

    def test_dataset(self):
        x = [10 + np.random.rand() for _ in range(40)]
        y = [3 * x for x in x]
        d = DataFrame(np.array([x, y]).T, columns=['x', 'y'])
        self.model.load_dataset(d)
        cov = self.model.mx_cov
        m, n = cov.shape
        self.assertEquals(m, 2)
        self.assertEquals(n, 2)
        for i, j in np.nditer(np.triu_indices(m)):
            if i == j:
                self.assertTrue(cov[i, i] > 0)
            else:
                self.assertEquals(cov[i, j], cov[j, i])
        self.assertTrue(np.all(np.linalg.eigvals(cov) > 0))

class test_model_pd_case(unittest.TestCase):

    def setUp(self):
        mod = pd.get_model()
        self.model = model.Model(mod)

    def test_size(self):
        n, m = self.model.mx_beta.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)
        n, m = self.model.mx_psi.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)
        n, m = self.model.mx_lambda.shape
        self.assertEqual(n, 2)
        self.assertEqual(m, 2)

    def test_variables(self):
        self.assertEquals(len(self.model.vars['Latents']), 3)
        self.assertEquals(len(self.model.vars['All']), 11)
        self.assertTrue('ind60' in self.model.vars['LatExo'])
        self.assertTrue('dem65' in self.model.vars['LatEndo'])
        self.assertTrue('dem60' in self.model.vars['LatEndo'])
        self.assertTrue('dem65' in self.model.vars['Outputs'])

    def test_dataset(self):
        data = pd.get_data()
        self.model.load_dataset(data)
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
