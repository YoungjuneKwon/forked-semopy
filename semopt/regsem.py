from .utils import get_cv_data_ann_kfold as get_cv_data
from .optimizer import Optimizer as Opt
from .stats import calculate_p_values
from collections import namedtuple
from .full_model import FullModel
from .model import Model
import numpy as np

TestResult = namedtuple('TestResult', ['ml_trn', 'ml_cv_mean', 'ml_cv',
                                       'pvals_nums', 'pvals'])


class StructureAnalyzer(object):
    def __init__(self, observed_variables: list, model_desc: str, data):
        """
        Keyword arguments:
            observed-variables -- A list of observed variables whose
                                  relationships are unclear.
            model_desc         -- A model description in a valid syntax,
                                  usually assumed to be a measurement part,
                                  but extra information (given caution)
                                  can be provided.
            data                - A data.
        """
        self.observed_vars = observed_variables
        self.model_desc = model_desc
        self.full_data = data.copy()
        self.training_set, self.testing_sets = get_cv_data(data)
        self.training_set = self.full_data
        self.testing_sets = [self.full_data]

    def get_model(self, params_to_ignore):
        model = FullModel(self.observed_vars, self.model_desc,
                          params_to_ignore)
        model.load_dataset(self.full_data)
        return model

    def test_model_cv(self, mod):
        mls = list()
        model = Model(self.get_model_description(mod),
                      force_load=self.observed_vars)
        opt = Opt(model)
        opt.load_dataset(self.training_set)
        lf = opt.optimize()
        pvals = np.array(calculate_p_values(opt))
        pvals = pvals[list(range(*model.beta_range))]
        pvals_nums = np.count_nonzero((pvals > 1e-1) | np.isnan(pvals))
        for data in self.testing_sets:
            data = data[model.vars['IndsObs']]
            cov = np.cov(data, rowvar=False, bias=True)
            opt.mx_cov = cov
            mls.append(opt.ml_wishart(opt.params))
        return TestResult(lf, np.mean(mls), mls, pvals_nums, sum(pvals))

    def get_least_significant_param(self, pvalues, params_to_pen, model, opt):
        pvalues = np.array(pvalues)
        t = pvalues[params_to_pen]
        if not len(t):
            return
        t = np.max(t)
        if np.isnan(t):
            t = np.array([True if i in params_to_pen else False
                          for i in range(len(pvalues))])
            i = np.where(np.isnan(pvalues) & t)[0][0]
        else:
            i = np.where(pvalues == t)[0][0]
        m, n = model.parameters['Beta'][i - model.beta_range[0]]
        lval, rval = model.beta_names[0][m], model.beta_names[1][n]
        return lval, rval

    def get_model_description(self, model):
        d = model.description
        op = model.operations.REGRESSION
        s = str()
        for lv in d:
            if d[lv][op]:
                s += '{} ~ {}\n'.format(lv, ' + '.join(list(d[lv][op].keys())))
        op = model.operations.MEASUREMENT
        for lv in d:
            if d[lv][op]:
                s += '{} =~ {}\n'.format(lv, ' + '.join(list(d[lv][op].keys())))
        return s

    def run(self, a: float, b: float, step: float, regu='l2'):
        params_to_ignore = set()
        for alpha in np.arange(a, b, step):
            model = self.get_model(params_to_ignore)
            params_to_pen = list(range(*model.beta_range))
            opt = Opt(model)
            lf = opt.optimize()
            pvalues = calculate_p_values(opt)
            ind = self.get_least_significant_param(pvalues, params_to_pen,
                                                   model, opt)
            params_to_ignore.add(ind)
            desc = self.get_model_description(model)
            yield (alpha, lf, 0, self.test_model_cv(model), desc)
            print(len(params_to_pen), lf)
            if len(params_to_pen) < 2:
                break

    def analyse(self, a: float, b: float, step: float, regu='l2',
                pval_cutoff=0.1):
        alphas, lfs, rfs, descs = list(), list(), list(), list()
        ml_trns, ml_means, ml_cvs = list(), list(), list()
        pvals_nums, pvals = list(), list()
        for alpha, lf, rf, test, desc in self.run(a, b, step, regu):
            alphas.append(alpha)
            lfs.append(lf)
            rfs.append(rf)
            ml_trns.append(test.ml_trn)
            ml_means.append(test.ml_cv_mean)
            ml_cvs.append(test.ml_cv)
            pvals_nums.append(test.pvals_nums)
            pvals.append(test.pvals)
            descs.append(desc)
        ml_cvs = np.array(ml_cvs).T
        return alphas, lfs, rfs, ml_trns, ml_means, ml_cvs, pvals_nums, pvals, descs
