"""
IEOR 262B MATHEMATICAL PROGRAMMING II
SUBSET SELECTION PROJECT

Author: Miguel Fernandez Montes

Dataset creation functions.

Both synthetic and real datasets are used.
"""


import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split


def normalize(X):
    """
    Transform the input matrix X into a matrix
    where the columns have unit L2-norm

    :param X: a numpy ndarray
    :return: an L2-normalized version of X

    >>> normalize(np.array([[3, 8], [4, 6]]))[0, :]
    array([0.6, 0.8])
    """
    return X / np.linalg.norm(X, axis=0)


def create_synthetic_dataset(n, p, k0, snr, corr=0, beta_type=1):
    """
    Create a synthetic regression dataset with the given parameters.

    :param n: number of rows or observations
    :param p: number of columns or features
    :param k0: number of true non-zero coefficients
    :param snr: signal-to-noise ratio
    :param corr: correlation factor
    :param beta_type: structure of the coefficient vector
        beta_type == 1: first k0 parameters are set to 1, the rest are set to zero
        beta_type == 2: equally spaced k0 parameters are set to 1, the rest are set
            to zero
        beta_type == 3: first k0 parameters are set to values equally spaced in
            the range [10, 0.5]
        beta_type == 5:
    :return:
        X: design matrix
        y: response vector
        beta: true parameter vector according to sparsity pattern
        cov: covariance matrix
        var: variance (noise term used to construct the response vector)
    """
    means = np.zeros(p)
    aux = np.abs(np.stack([np.arange(-j, p - j) for j in range(p)], axis=0))
    cov = np.power(corr, aux)

    if beta_type == 1:
        beta = np.concatenate([np.ones(k0), np.zeros(p - k0)])
    elif beta_type == 2:
        space = np.ceil(p / k0).astype(int)
        beta = np.array([1 if ((i % space) == 0) else 0 for i in range(p)])
    elif beta_type == 3:
        beta = np.concatenate([np.linspace(10, 0.5, k0), np.zeros(p - k0)])
    elif beta_type == 5:
        beta = np.concatenate([np.ones(k0),
                               np.array([0.5 ** (i + 1 - k0)
                                         for i in range(k0, p)])])
    else:
        raise NotImplementedError

    X = np.random.multivariate_normal(mean=means, cov=cov, size=n)
    # X = normalize(X)  # for now i dont normalize

    var = (beta.T @ cov @ beta) / snr
    y = X @ beta + np.random.normal(loc=0, scale=np.sqrt(var), size=n)

    return X, y, beta, cov, var


def load_dataset(name, data_dir, k0=10, snr=5):
    """
    Load dataset and return design matrix and response

    :param name: name of dataset
    :param data_dir: path to data directory
    :param k0: number of true nonzero parameters
    :param snr: signal-to-noise ratio
    :return:
        X_train: training desing matrix
        y_train: training response vector
        X_test: test matrix
        y_test: test response vector
        beta: true parameter vector
        var: variance (noise term used to construct the response)
    """
    if name == 'prostate':
        prostate = pd.read_csv(os.path.join(data_dir, 'prostate.csv'))
        prostate = prostate.drop(columns='Unnamed: 0')
        X_orig, y_orig = prostate.iloc[:, :-1].values, prostate.iloc[:, -1].values

    elif name == 'lymphoma':
        lymphoma = pd.read_csv(os.path.join(data_dir, 'prostate.csv'))
        lymphoma = lymphoma.drop(columns='Unnamed: 0')
        X_orig, y_orig = lymphoma.iloc[:, :-1].values, lymphoma.iloc[:, -1].values

    else:
        raise NotImplementedError

    X_centered = X_orig - X_orig.mean(axis=0)

    selector = SelectKBest(k=1000)

    X = selector.fit_transform(X_centered, y_orig)
    n, p = X.shape
    beta = np.concatenate([np.ones(k0), np.zeros(p - k0)])
    var = np.var(X @ beta) / snr
    y = X @ beta + np.random.normal(loc=0, scale=np.sqrt(var), size=n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=5)

    return X_train, y_train, X_test, y_test, beta, var
