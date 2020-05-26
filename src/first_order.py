"""
IEOR 262B MATHEMATICAL PROGRAMMING II
SUBSET SELECTION PROJECT

Author: Miguel Fernandez Montes

Implementation of the discrete first-order method
for sparse linear regression proposed by
Bertsimas et al. (2016)

Note: the algorithm may implement a line search to find the next
iteration coefficient but for now only the original algorithm
has been employed
"""


import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import lstsq
import datasets as ds


def threshold(arr, k):
    """
    Threshold array to its top k elements (by absolute value)
    Equivalent to the operator H_k described by Bertsimas et al.

    :param arr: input array
    :param k: number of elements to keep
    :return: thresholded array
    """
    idx = np.argpartition(np.abs(arr), -k)[-k:]
    arr_thresh = np.zeros_like(arr)
    arr_thresh[idx] = arr[idx]
    return arr_thresh


def ls_obj(X, y, beta):
    return (1 / 2) * np.sum((y - X @ beta) ** 2)


def ls_grad(X, y, beta):
    return -X.T @ (y - X @ beta)


def find_best_step(X, y, beta, eta):
    # line_search_results = line_search(lambda s: ls_obj(X, y, s),
    #                                   lambda s: ls_grad(X, y, s),
    #                                   beta, eta - beta)
    # step = line_search_results[0]

    res = minimize_scalar(lambda s: ls_obj(X, y, s * eta + (1 - s) * beta),
                          bounds=(0, 1), method='bounded')
    lmbda = res.x
    # print(f'Step min scalar {lmbda}')
    # print(f'Minimize scalar {res.fun}')
    return lmbda


def discrete_first_order(X, y, k, beta_init=None, max_iter=1000, tol=1e-04,
                         polish=True, search=False):
    """
    Compute a sparse regression parameter vector using
    the discrete first-order method by Bertsimas et al.

    :param X: design matrix
    :param y: response
    :param k: constraints on number of nonzero coefficients
    :param beta_init: initial value of beta
    :param max_iter:
    :param tol:
    :param polish:
    :param search: if True, perform line search
    :return: beta: parameter vector
            obj: value of the objective function
    """
    n, p = X.shape
    L = np.real(np.max(np.linalg.eigvals(X.T @ X)))  # TODO approximate with power method

    if beta_init is None:
        # randomly initialize beta
        init_means = np.zeros(p)
        init_cov = 4 * np.identity(p)
        draws = np.random.multivariate_normal(init_means, init_cov)
        beta = threshold(draws, k)
    else:
        beta = beta_init

    last_obj = ls_obj(X, y, beta)

    # iterations
    for t in range(max_iter):
        if search:
            # the implementation with line search is not complete
            pass
        else:
            beta = threshold(beta - (1 / L) * ls_grad(X, y, beta), k)

        obj = ls_obj(X, y, beta)

        if np.abs(last_obj - obj) < tol:
            break

        last_obj = obj

    if polish:
        # solve OLS on active set
        if search:
            # the implementation with line search is not complete
            pass
        else:
            active_idx = (beta != 0)

        X_A = X[:, active_idx]

        beta_polish_A, _, _, _ = lstsq(X_A, y)
        beta_polish = np.zeros(p)
        beta_polish[active_idx] = beta_polish_A

        beta = beta_polish
        obj = ls_obj(X, y, beta)

    return beta, obj


def best_subset_first_order(X, y, k, ls_init=True, max_iter=1000, tol=1e-04,
                            polish=True):
    """
    Run the discrete first-order method described by Bertsimas et al.
    for 50 random initializations of the coefficient vector.

    :param X: design matrix
    :param y: response
    :param k: constraint on number of nonzero coefficients
    :param ls_init: whether to initialize beta using thresholded LS solution
    :param max_iter: maximum number of iterations for the algorithm
    :param tol: tolerance
    :param polish: if True, compute OLS coefficients on active set
        at the end of the algorithm
    :return:
    """
    n, p = X.shape

    if ls_init:
        if p < n:
            beta_ls, _, _, _ = lstsq(X, y)
            beta_ls = threshold(beta_ls, k)
        else:
            beta_ls = threshold(X.T @ y, k)
        beta_init = beta_ls
    else:
        beta_init = None

    best_obj = float('inf')
    best_beta = beta_init

    for i in range(50):
        beta, obj = discrete_first_order(X, y, k, beta_init=beta_init,
                                         max_iter=max_iter,
                                         tol=tol,
                                         polish=polish)

        if obj < best_obj:
            best_beta = beta
            best_obj = obj

        # randomized next initial beta from Tibshirani's R code
        if ls_init:
            beta_init = beta_ls + 2 * np.random.rand(p) * np.max(np.abs(beta_ls))

    return best_beta


if __name__ == '__main__':
    # test
    np.random.seed(42)
    X, y, beta0, _, _ = ds.create_synthetic_dataset(50, 1000, 5, 1, beta_type=1)

    beta = best_subset_first_order(X, y, 5, ls_init=True)
