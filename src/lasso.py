"""
IEOR 262B MATHEMATICAL PROGRAMMING II
SUBSET SELECTION PROJECT

Author: Miguel Fernandez Montes

Implementation of Lasso and relaxed Lasso for linear regression
"""


import numpy as np
from sklearn.linear_model import Lasso, LassoLars
from scipy.linalg import lstsq
import datasets as ds


def best_subset_lasso(X, y, lmbda):
    """
    Instantiate a scikit-learn Lasso regression model
    and fit it to the data X, y

    :param X: design matrix
    :param y: response
    :param lmbda: regularization parameter
    :return: model.coef_: the parameter estimate vector
    """
    n, p = X.shape
    if n > p:
        model = Lasso(alpha=lmbda, fit_intercept=False)
        model.fit(X, y)
    else:
        model = LassoLars(alpha=lmbda, fit_intercept=False)
        model.fit(X, y)
    return model.coef_


def best_subset_relaxed_lasso(X, y, lmbda, gamma,
                              compute_lasso=True,
                              beta_lasso=None):
    """
    Compute the relaxed Lasso regression coefficient vector
    for a given set of parameters lmbda and gamma

    :param X: design matrix
    :param y: response
    :param lmbda: Lasso regularization parameter
    :param gamma: parameter regulating the tradeoff between
                the Lasso solution and the LS solution
    :param compute_lasso: if True, computes the
    :param beta_lasso: solution of the Lasso
    :return: beta: the relaxed Lasso parameter estimate vector
    """
    if compute_lasso or beta_lasso is None:
        beta_lasso = best_subset_lasso(X, y, lmbda)

    active_idx = (beta_lasso != 0)
    X_A = X[:, active_idx]

    beta_ls_A, _, _, _ = lstsq(X_A, y)
    beta_ls = np.zeros_like(beta_lasso)
    beta_ls[active_idx] = beta_ls_A

    beta = gamma * beta_lasso + (1 - gamma) * beta_ls
    return beta


if __name__ == '__main__':
    # test
    X, y, beta0, _, _ = ds.create_synthetic_dataset(50, 1000, 5, 0.48, beta_type=1)
    beta_lasso = best_subset_lasso(X, y, 0.005)
    print("Lasso:\n")
    print(beta_lasso)

    beta_rlasso = best_subset_relaxed_lasso(X, y, 0.005, 1,
                                            compute_lasso=False,
                                            beta_lasso=beta_lasso)
    print("Relaxed Lasso:\n")
    print(beta_rlasso)
