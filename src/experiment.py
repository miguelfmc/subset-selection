"""
IEOR 262B MATHEMATICAL PROGRAMMING II
SUBSET SELECTION PROJECT

Author: Miguel Fernandez Montes

Script to run an experiment for a specific dataset,
computing the discrete first-order, best subset, lasso
and relaxed lasso solutions

The process follows the steps outlined in Hastie et al. (2017)

1. Create synthetic data sets (both train and validation)
    with a specific beta-type, n, p, k0,
    correlation factor and SNR

2. Compute best subset, 1st order, lasso and relaxed lasso
    for different tuning parameters
    Choose the model that yields lower MSE on the validation set

3. Record evaluation metrics
    Relative risk
    Relative test error
    Proportion of variance explained
    Number of zeros
    Relative parameter vector error
    Missed detection rate
    False alarm rate

4. Repeat 10 times (with 10 pairs of data sets) and average metrics results
"""


import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datasets as ds
from lasso import best_subset_lasso, best_subset_relaxed_lasso
from mio import best_subset_mio
from first_order import best_subset_first_order
from metrics import relative_risk, relative_test_error, proportion_var_explained
from metrics import relative_beta_error, missed_detection_rate, false_alarm_rate
from metrics import mse, osr2
import config


# Names and directories
# These should be set according to your local setup
experiments_path = config.experiments_path
data_path = config.data_path


def generate_name(setting, beta_type, corr, snr):
    """
    Create file name from experiment characteristics

    :param setting: dimensionality type of the dataset
    :param beta_type: sparsity pattern
    :param corr: correlation factor
    :param snr: signal-to-noise ratio
    :return: name: file name
    """
    name = setting + '_beta_' + str(beta_type) + '_corr_' + str(corr) + '_snr_' + str(snr)
    return name


def run_experiment(setting, beta_type, corr, snr, n_runs=10, fast=True):
    """
    Run experiment on artificial dataset with Lasso, Relaxed Lasso, 1st order, MIO
    and record metrics to a CSV file

    :param setting: one of low, mid, high
    :param beta_type: sparsity pattern
    :param corr: correlation factor
    :param snr: signal-to-noise ratio
    :param n_runs: number of data sets on which to run the methods
    :param fast: if True, don't validate MIO and use best k from first order
    """
    if setting == 'low':
        n, p, k0 = 100, 10, 5
    elif setting == 'mid':
        n, p, k0 = 500, 100, 5
    elif setting == 'high':
        n, p, k0 = 50, 1000, 5
    else:
        raise NotImplementedError

    experiment_name = generate_name(setting, beta_type, corr, snr)
    runs = []
    times = []
    column_names = ['setting', 'beta_type', 'corr', 'snr',
                    'run', 'method', 'RR', 'RTE', 'PVE',
                    'non_zeros', 'beta_error', 'MDR', 'FAR']

    for i in range(n_runs):
        # 1. Create data sets
        X, y, beta0, cov, var = ds.create_synthetic_dataset(n, p, k0, snr, corr, beta_type)
        X_val, y_val, *_ = ds.create_synthetic_dataset(n, p, k0, snr, corr, beta_type)

        # 2. Models
        # Lasso and Relaxed Lasso
        # Since relaxed Lasso uses the Lasso solution we run the validation scheme
        # at the same time for both methods
        lmbda_max = np.max(np.abs(X.T @ y))
        lmbda_min = 1e-6 * lmbda_max  # default glmnet min ratio is 1e-06

        if n > p:
            num_lmbdas = 50
        else:
            num_lmbdas = 100

        lmbdas = np.geomspace(lmbda_min, lmbda_max, num_lmbdas)
        gammas = np.linspace(0, 1, 10)
        # initialize betas and mses
        beta_lasso_best = np.zeros(p)
        beta_rlasso_best = np.zeros(p)
        mse_lasso_best = float('inf')
        mse_rlasso_best = float('inf')

        start = time.time()
        for lmbda in lmbdas:
            beta_lasso = best_subset_lasso(X, y, lmbda)
            for gamma in gammas:
                beta_rlasso = best_subset_relaxed_lasso(X, y, lmbda, gamma,
                                                        compute_lasso=False,
                                                        beta_lasso=beta_lasso)
                mse_rlasso = mse(y_val, X_val @ beta_rlasso)
                if mse_rlasso < mse_rlasso_best:
                    beta_rlasso_best = beta_rlasso
                    mse_rlasso_best = mse_rlasso

            mse_lasso = mse(y_val, X_val @ beta_lasso)
            if mse_lasso < mse_lasso_best:
                beta_lasso_best = beta_lasso
                mse_lasso_best = mse_lasso
        end = time.time()
        time_lasso = end - start

        # Discrete first order and MIO
        # Since the MIO method uses warm-starts with the first-order solution
        # we run their validation in parallel
        if n > p:
            num_ks = 10
        else:
            num_ks = 10

        ks = np.arange(1, num_ks + 1)

        # initialize betas and mses
        beta_mio_best = np.zeros(p)
        beta_first_order_best = np.zeros(p)
        mse_mio_best = float('inf')
        mse_first_order_best = float('inf')

        if fast:
            # only validate first order
            start = time.time()
            for k in ks:
                beta_first_order = best_subset_first_order(X, y, k)
                mse_first_order = mse(y_val, X_val @ beta_first_order)

                if mse_first_order < mse_first_order_best:
                    beta_first_order_best = beta_first_order
                    mse_first_order_best = mse_first_order
                    best_k = k

            beta_mio_best, _ = best_subset_mio(X, y, best_k,
                                               beta_first_order=beta_first_order,
                                               warm_start=True, time_limit=180)
            end = time.time()
            time_mio = end - start

        else:
            # validate first order and MIO
            start = time.time()
            for k in ks:
                beta_first_order = best_subset_first_order(X, y, k)
                beta_mio, _ = best_subset_mio(X, y, k,
                                              beta_first_order=beta_first_order,
                                              warm_start=True, time_limit=180)
                mse_first_order = mse(y_val, X_val @ beta_first_order)
                mse_mio = mse(y_val, X_val @ beta_mio)

                if mse_first_order < mse_first_order_best:
                    beta_first_order_best = beta_first_order
                    mse_first_order_best = mse_first_order

                if mse_mio < mse_mio_best:
                    beta_mio_best = beta_mio
                    mse_mio_best = mse_mio
            end = time.time()
            time_mio = end - start

        # 3. Evaluation metrics from best models
        # Save parameters from each model
        results = {'lasso': beta_lasso_best,
                   'rlasso': beta_rlasso_best,
                   'first_order': beta_first_order_best,
                   'mio': beta_mio_best}

        rows = []
        for method, beta in results.items():
            rr = relative_risk(beta, beta0, cov)
            rte = relative_test_error(beta, beta0, cov, var)
            pve = proportion_var_explained(beta, beta0, cov, var)
            num_non_zeros = (beta != 0).sum()
            rbe = relative_beta_error(beta, beta0)
            mdr = missed_detection_rate(beta, beta0)
            far = false_alarm_rate(beta, beta0)
            model_metrics = [setting, beta_type, corr, snr,
                             i, method,
                             rr, rte, pve, num_non_zeros,
                             rbe, mdr, far]
            # write it out
            rows.append(model_metrics)

        run_results = pd.DataFrame(data=rows, columns=column_names)
        runs.append(run_results)

        run_times = pd.DataFrame(data=[[setting, beta_type, corr, snr,
                                        i, time_lasso, time_mio]],
                                 columns=['setting', 'beta_type', 'corr', 'snr',
                                          'run', 'time_lasso', 'time_mio'])
        times.append(run_times)

    experiment_results = pd.concat(runs)
    experiment_results.to_csv(os.path.join(experiments_path, experiment_name + '.csv'))

    experiment_times = pd.concat(times)
    experiment_times.to_csv((os.path.join(experiments_path, experiment_name + '_times.csv')))


def run_experiment_real(dataset_name, fast=True):
    """
    Run experiments on real datasets

    :param dataset_name: name of the real dataset
    :param fast: if True, validate k with first order method only
    """
    experiment_name = dataset_name
    column_names = ['dataset', 'method', 'test_mse', 'OSR2',
                    'non_zeros', 'beta_error', 'MDR', 'FAR']

    # 1. Load dataset
    # TODO implement cross validation
    X_train, y_train, X_test, y_test, beta0, var = ds.load_dataset(dataset_name, data_dir=data_path)
    X, X_val, y, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=10)

    n, p = X.shape

    # 2. Models
    # Lasso and Relaxed Lasso
    # Since relaxed Lasso uses the Lasso solution we run the validation scheme
    # at the same time for both methods
    lmbda_max = np.max(np.abs(X.T @ y))
    lmbda_min = 1e-6 * lmbda_max  # default glmnet min ratio is 1e-06

    num_lmbdas = 100

    lmbdas = np.geomspace(lmbda_min, lmbda_max, num_lmbdas)
    gammas = np.linspace(0, 1, 10)
    # initialize betas and mses
    beta_lasso_best = np.zeros(p)
    beta_rlasso_best = np.zeros(p)
    mse_lasso_best = float('inf')
    mse_rlasso_best = float('inf')

    start = time.time()
    for lmbda in lmbdas:
        beta_lasso = best_subset_lasso(X, y, lmbda)
        for gamma in gammas:
            beta_rlasso = best_subset_relaxed_lasso(X, y, lmbda, gamma,
                                                    compute_lasso=False,
                                                    beta_lasso=beta_lasso)
            mse_rlasso = mse(y_val, X_val @ beta_rlasso)
            if mse_rlasso < mse_rlasso_best:
                beta_rlasso_best = beta_rlasso
                mse_rlasso_best = mse_rlasso

        mse_lasso = mse(y_val, X_val @ beta_lasso)
        if mse_lasso < mse_lasso_best:
            beta_lasso_best = beta_lasso
            mse_lasso_best = mse_lasso
    end = time.time()
    time_lasso = end - start


    # Discrete first order and MIO
    # Since the MIO method uses warm-starts with the first-order solution
    # we run their validation in parallel
    num_ks = 20

    ks = np.arange(1, num_ks + 1)

    # initialize betas and mses
    beta_mio_best = np.zeros(p)
    beta_first_order_best = np.zeros(p)
    mse_mio_best = float('inf')
    mse_first_order_best = float('inf')

    if fast:
        # only validate first order
        start = time.time()
        for k in ks:
            beta_first_order = best_subset_first_order(X, y, k)
            mse_first_order = mse(y_val, X_val @ beta_first_order)

            if mse_first_order < mse_first_order_best:
                beta_first_order_best = beta_first_order
                mse_first_order_best = mse_first_order
                best_k = k

        beta_mio_best, _ = best_subset_mio(X, y, best_k,
                                           beta_first_order=beta_first_order,
                                           warm_start=True, time_limit=720)  # more time for Gurobi
        end = time.time()
        time_mio = end - start

    else:
        # validate first order and MIO
        start = time.time()
        for k in ks:
            beta_first_order = best_subset_first_order(X, y, k)
            beta_mio, _ = best_subset_mio(X, y, k,
                                          beta_first_order=beta_first_order,
                                          warm_start=True, time_limit=180)
            mse_first_order = mse(y_val, X_val @ beta_first_order)
            mse_mio = mse(y_val, X_val @ beta_mio)

            if mse_first_order < mse_first_order_best:
                beta_first_order_best = beta_first_order
                mse_first_order_best = mse_first_order

            if mse_mio < mse_mio_best:
                beta_mio_best = beta_mio
                mse_mio_best = mse_mio
        end = time.time()
        time_mio = end - start

    # 3. Evaluation metrics from best models
    # Save parameters from each model
    results = {'lasso': beta_lasso_best,
               'rlasso': beta_rlasso_best,
               'first_order': beta_first_order_best,
               'mio': beta_mio_best}

    rows = []
    for method, beta in results.items():
        test_mse = mse(y_test, X_test @ beta)
        test_acc = osr2(y_test, X_test @ beta)
        num_non_zeros = (beta != 0).sum()
        rbe = relative_beta_error(beta, beta0)
        mdr = missed_detection_rate(beta, beta0)
        far = false_alarm_rate(beta, beta0)
        model_metrics = [dataset_name, method,
                         test_mse, test_acc, num_non_zeros,
                         rbe, mdr, far]
        # write it out
        rows.append(model_metrics)

    results = pd.DataFrame(data=rows, columns=column_names)

    times = pd.DataFrame(data=[[dataset_name, time_lasso, time_mio]],
                         columns=['dataset', 'time_lasso', 'time_mio'])

    results.to_csv(os.path.join(experiments_path, experiment_name + '.csv'))

    times.to_csv((os.path.join(experiments_path, experiment_name + '_times.csv')))


def main():
    # test
    print('This is a test experiment!')
    run_experiment('mid', 1, 0, 3, n_runs=10)


if __name__ == '__main__':
    main()
