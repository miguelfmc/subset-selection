"""
IEOR 262B MATHEMATICAL PROGRAMMING II
SUBSET SELECTION PROJECT

Author: Miguel Fernandez Montes

Implementation of the best subset selection for linear regression
as a mixed integer optimization problem based on the work by
Bertsimas et al. (2016)

The problem is formulated as:

    min (1/2) * || y - X beta ||^2
    s.t. || beta ||_0  <= k

For n > p (low dimensional design matrix) the formulation used is:

...

For p > n (high dimensional design matrix) the formulation used is:

...

It is a mixed integer quadratic program with linear and
SOS-1 constraints.

The bounds on beta and || beta ||_1 are either given by ...

The problem can be optimized with "warm-starts", taking
the (approximate) solution given by a discrete 1st order method
as initial value of beta.
"""


import numpy as np
import gurobipy as gp
from gurobipy import GRB
import datasets as ds
from first_order import best_subset_first_order


# could also use cumulative coherence and restricted eigenvalue
# TODO implement theoretical bounds
def beta_l1_bound(X, y, mu, k):
    pass


def beta_linf_bound(X, y, gamma):
    pass


def zeta_l1_bound(X, y, beta_l1, k):
    pass


def zeta_linf_bound(X, beta_linf, k):
    aux = X.copy()
    tmp = np.sort(np.abs(aux), axis=1)[:, -k:].sum(axis=1).max()
    return tmp * beta_linf


def get_bounds(X, y, k):
    mu = np.max(np.abs(X.T @ X))
    gamma = np.min(np.linalg.eigvals(X.T @ X))

    beta_l1 = beta_l1_bound(X, y, mu, k)
    beta_linf = beta_linf_bound(X, y, gamma)
    zeta_l1 = zeta_l1_bound(X, y, beta_l1, k)
    zeta_linf = zeta_linf_bound(X, beta_linf, k)

    return beta_l1, beta_linf, zeta_l1, zeta_linf


def best_subset_mio(X, y, k, warm_start=False, beta_first_order=None, tau=2, time_limit=GRB.INFINITY):
    """
    Compute the best subset solution for linear regression by solving a
    Mixed Integer Optimization problem

    :param X: design matrix
    :param y: response
    :param k: sparsity constraint
    :param beta_first_order: solution from first-order method
    :param warm_start: if True, warm-start using 1st order method solution
    :param tau: parameter to define an upper bound on beta
    :param time_limit: time limit for solver
    :return: beta.X: the parameter estimate vector
            model: the Gurobi model instance
    """
    n, p = X.shape

    if k == 0:
        # trivial case
        beta = np.zeros(p)
        return beta, None

    if warm_start and beta_first_order is not None:
        print('Warm start!')
        beta_init = beta_first_order.copy()
        z_init = (beta_init != 0).astype(int)
        M_U = tau * np.max(np.abs(beta_init))
    else:
        beta_init = np.zeros(p)
        z_init = np.zeros(p)
        M_U = 10  # TODO fix temporary solution, get theoretical bounds!

    if n >= p:
        # Create a new model
        model = gp.Model('MIO')

        # Set time limit
        model.Params.TimeLimit = time_limit

        # Set quiet output
        model.Params.OutputFlag = 0

        # Create variables
        beta = model.addMVar(shape=p,
                             lb=-M_U,
                             ub=M_U,
                             vtype=GRB.CONTINUOUS,  # not sure about this
                             name='beta')
        beta.Start = beta_init

        z = model.addMVar(shape=p,
                          vtype=GRB.BINARY,
                          lb=0,
                          ub=1,
                          name='z')
        z.Start = z_init

        # Set objective
        obj = beta @ (X.T @ X) @ beta - 2 * (X.T @ y) @ beta + y @ y
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints - formulated without specially ordered sets
        model.addConstr(beta <= M_U * z, name='constrain_upper')
        model.addConstr(beta >= -M_U * z, name='constraint_lower')
        model.addConstr(z.sum() <= k, name='sparsity_constraint')

        # TODO Constraints - formulated with specially ordered sets
        # TODO l1 norm constraint
        # Hastie et al. don't seem to upper bound the L1 norm of
        # the coefficients either so maybe it doesn't matter

        # Optimize
        model.optimize()

    else:
        # Create a new model
        model = gp.Model('MIO')

        # Set time limit
        model.Params.TimeLimit = time_limit

        # Set quiet output
        model.Params.OutputFlag = 0

        # Create variables
        beta = model.addMVar(shape=p,
                             lb=-M_U,
                             ub=M_U,
                             vtype=GRB.CONTINUOUS,  # not sure about this
                             name='beta')
        beta.Start = beta_init

        z = model.addMVar(shape=p,
                          vtype=GRB.BINARY,
                          lb=0,
                          ub=1,
                          name='z')
        z.Start = z_init

        # zeta aux variable
        M_U_zeta = zeta_linf_bound(X, np.max(np.abs(beta_init)), k)
        zeta_init = X @ beta_init
        zeta = model.addMVar(shape=n,
                             lb=-M_U_zeta,
                             ub=M_U_zeta,
                             vtype=GRB.CONTINUOUS,
                             name='zeta')
        zeta.Start = zeta_init

        # Set objective
        obj = zeta @ zeta - 2 * (X.T @ y) @ beta + y @ y
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints - formulated without specially ordered sets
        model.addConstr(beta <= M_U * z, name='constrain_upper')
        model.addConstr(beta >= -M_U * z, name='constraint_lower')
        model.addConstr(z.sum() <= k, name='sparsity_constraint')
        model.addConstr(zeta == X @ beta, name='zeta_constraint')

        # Optimize
        model.optimize()

    return beta.X, model


if __name__ == '__main__':
    # test
    np.random.seed(42)
    X, y, beta0, _, _ = ds.create_synthetic_dataset(50, 1000, 5, 1, corr=0.35)
    beta_first_order = best_subset_first_order(X, y, 5)
    beta, model = best_subset_mio(X, y, 5, warm_start=True,
                                  beta_first_order=beta_first_order,
                                  time_limit=180)