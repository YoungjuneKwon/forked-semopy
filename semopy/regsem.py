from .utils import get_cv_data_ann_kfold as get_cv_data
from .optimizer import Optimizer as Opt
from .stats import calculate_p_values
from collections import namedtuple
from .full_model import FullModel
from .model import Model
import numpy as np

TestResult = namedtuple('TestResult', ['ml_trn', 'ml_cv_mean', 'ml_cv'])


class StructureAnalyzer(object):
    def __init__(self, observed_variables: list, model_desc: str, data,
                 use_cv=True):
        """
        Keyword arguments:
        observed-variables -- A list of observed variables whose
                              relationships are unclear.
        model_desc         -- A model description in a valid syntax,
                              usually assumed to be a measurement part,
                              but extra information (given caution)
                              can be provided.
        data               -- A data.
        use_cv             -- Use cross-validation.
        """
        self.observed_vars = observed_variables
        self.model_desc = model_desc
        self.full_data = data.copy()
        self.training_set, self.testing_sets = get_cv_data(data)
        if not use_cv:
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
#        pvals = np.zeros((model.beta_range[1] - model.beta_range[0],))
        for data in self.testing_sets:
            data = data[model.vars['IndsObs']]
            cov = np.cov(data, rowvar=False, bias=True)
            opt.mx_cov = cov
            mls.append(opt.ml_wishart(opt.params))
#        pvals /= len(self.testing_sets)
#        pvals_nums = np.count_nonzero((pvals > 1e-1) | np.isnan(pvals))
        return TestResult(lf, np.mean(mls), mls)

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

    def get_num_pvals(self, opt, pvals):
        pvals = np.array(pvals)
        return np.count_nonzero(pvals[list(range(*opt.model.beta_range))] > 5e-2)

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

    def run(self):
        params_to_ignore = set()
        while True:
            model = self.get_model(params_to_ignore)
            params_to_pen = list(range(*model.beta_range))
            opt = Opt(model)
            lf = opt.optimize()
            conn = self.get_num_components_connected(opt)
            if conn < 6:
                break
            pvalues = calculate_p_values(opt)
            num_pvals = self.get_num_pvals(opt, pvalues)
            ind = self.get_least_significant_param(pvalues, params_to_pen,
                                                   model, opt)
            params_to_ignore.add(ind)
            if len(params_to_pen) > 2 * conn:
                lf = np.inf
            desc = self.get_model_description(model)
            yield lf, self.test_model_cv(model), num_pvals, conn, desc
            if len(params_to_pen) < 2:
                break

    def get_num_components_connected(self, opt):
        """Get number of variables present in structural part.

        Keyword arguments:
        opt -- Optimizer with optimized parameters.

        Returns:
        Number of variables
        """
        n = opt.mx_beta.shape[0]
        for i in range(n):
            r_nonzeros = np.abs(opt.mx_beta[i]) > 1e-16
            c_nonzeros = np.abs(opt.mx_beta[:, i]) > 1e-16
            num_nonzeros = r_nonzeros | c_nonzeros
            if not np.count_nonzero(num_nonzeros):
                n -= 1
        return n

    def analyze(self, print_status=False):
        """Wraps run method and returns helper structures.

        Keyword arguments:
        print_status -- Whether to print intermediate information on each step.

        Returns:
        Array of models numbers, MLs of FullModel, mean CV ML, CV MLs, numbers
        of p-values exceeding set bound, sums of pvalues, numbers of present
        in variables in structural part, models' descriptions
        """
        n, lfs, descs = list(), list(), list()
        ml_trns, ml_means, ml_cvs = list(), list(), list()
        pvals_nums, conns = list(), list()
        for i, (lf, test, num_pvals, conn, desc) in enumerate(self.run()):
            if lf is None or lf is np.nan:
                continue
            n.append(i)
            lfs.append(lf)
            conns.append(conn)
            ml_trns.append(test.ml_trn)
            ml_means.append(test.ml_cv_mean)
            ml_cvs.append(test.ml_cv)
            pvals_nums.append(num_pvals)
            descs.append(desc)
            if ml_means[-1] > 16:
                break
            if print_status:
                print("Step {}, {:.4f}, {:.4f}, pnum: {}".format(i,
                                                                 test.ml_cv_mean,
                                                                 lf, num_pvals))
        n, lfs, ml_cvs = np.array(n), np.array(lfs), np.array(ml_cvs).T
        ml_means, conns = np.array(ml_means), np.array(conns)
        pvals_nums = np.array(pvals_nums)
        return n, lfs, ml_trns, ml_means, ml_cvs, pvals_nums, conns, descs
