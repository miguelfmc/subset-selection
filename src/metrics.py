"""
IEOR 262B MATHEMATICAL PROGRAMMING II
SUBSET SELECTION PROJECT

Author: Miguel Fernandez Montes

Metrics for model evaluation
"""


import numpy as np


def relative_risk(beta_hat, beta0, cov):
    return (beta_hat - beta0).T @ cov @ (beta_hat - beta0) / (beta0 @ cov @ beta0)


def relative_test_error(beta_hat, beta0, cov, var):
    return ((beta_hat - beta0).T @ cov @ (beta_hat - beta0) + var) / var


def proportion_var_explained(beta_hat, beta0, cov, var):
    return 1 - \
           ((beta_hat - beta0).T @ cov @ (beta_hat - beta0) + var) / (beta0 @ cov @ beta0 + var)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def osr2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def relative_beta_error(beta_hat, beta0):
    return np.sum((beta_hat - beta0) ** 2) / np.sum(beta0 ** 2)


def missed_detection_rate(beta_hat, beta0):
    """
    Proportion of missed detections or coefficient estimates
    that are zero when the true parameters are nonzero.

    Adapted from Reeves et al. (2013)

    :param beta_hat: parameter estimates
    :param beta0: true parameters
    :return: missed detection rate
    """
    return np.mean((beta_hat == 0) & (beta0 != 0))


def false_alarm_rate(beta_hat, beta0):
    """
    Proportion of false alarms or coefficient estimates
    that are nonzero when the true parameters are zero

    From Reeves et al. (2013)

    :param beta_hat: parameter estimates
    :param beta0: true parameters
    :return: false alarm rate
    """
    return np.mean((beta_hat != 0) & (beta0 == 0))
