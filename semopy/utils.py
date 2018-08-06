import scipy.linalg.lapack as lapack
from pandas import DataFrame
import numpy as np


def cov(x: np.array, bias=True,):
    """Computes covariance matrix taking in account missing values.

    Key arguments:
    x    -- A DataFrame.
    bias -- Bias.

    Returns:
    Covariance matrix.
    """
    masked_x = np.ma.array(x, mask=np.isnan(x))
    return np.ma.cov(masked_x, bias=bias, rowvar=False).data


def chol_inv(x: np.array):
    """Calculates invserse of matrix using Cholesky decomposition.

    Keyword arguments:
    x -- A matrix.

    Returns:
    x^-1.
    """
    c, info = lapack.dpotrf(x)
    if info:
        raise np.linalg.LinAlgError
    lapack.dpotri(c, overwrite_c=1)
    return c + np.triu(c, 1).T


def get_cv_data_ann_kfold(data, k=4, iteration=1, shuffle=True):
    if shuffle:
        data = data.sample(frac=1)
    chunk_size = data.shape[0] // k
    training_set = None
    testing_sets = list()
    for i in range(k):
        a = i * chunk_size
        b = a + chunk_size
        if i == iteration:
            training_set = data[a:b]
        else:
            testing_sets.append(data[a:b])
    return training_set, testing_sets


def get_cv_data_kfold(data, k=4, iteration=1, shuffle=False):
    if shuffle:
        np.random.shuffle(data)
    chunk_size = data.shape[0] // k
    a = iteration * chunk_size
    b = a + chunk_size
    inds_training = np.r_[:a, b:]
    inds_testing = np.r_[a:b]
    if isinstance(data, DataFrame):
        training_set = data.loc[inds_training]
        testing_set = data.loc[inds_testing]
    else:
        training_set = data[inds_training]
        testing_set = data[inds_testing]
    return training_set, [testing_set]
