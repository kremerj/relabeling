# All selection criteria have to be maximized

import numpy as np
from scipy.stats import entropy
from noise import get_noise_matrix
from util import is_deep


def margin_selection(clf, Sx):
    return entropy(clf.predict_proba(Sx).T)


def agnostic_selection(clf, Sx, Sy_noise, rho, lx, q, solver_params):
    pred = clf.predict_proba(Sx)
    if is_deep(solver_params):
        df = clf.gradient_diff_norm(Sx, Sy_noise)
        K = get_noise_matrix(rho)
        return np.array([np.sum(p * K[:, y] * d) for p, y, d in zip(pred, Sy_noise, df)])
    else:
        return lx * np.array(
            [rho[1 - y] * p[1 - y] / (rho[1 - y] * p[1 - y] + (1 - rho[y]) * p[y]) for p, y in zip(pred, Sy_noise)])


def ml_selection(clf, Sx, Sy_noise, rho, lx, q, solver_params):
    pred = clf.predict_proba(Sx)
    if is_deep(solver_params):
        df = clf.robust_gradient_diff_norm(Sx, Sy_noise, q)
        K = get_noise_matrix(rho)
        return np.array([np.sum(p * K[:, y] * d) for p, y, d in zip(pred, Sy_noise, df)])
    else:
        return lx * np.array(
            [(1 - rho[y]) * rho[1 - y] * p[y] * p[1 - y] / (rho[1 - y] * p[1 - y] + (1 - rho[y]) * p[y]) ** 2. for p, y
             in zip(pred, Sy_noise)])


def ub_selection(clf, Sx, Sy_noise, rho, lx, q, solver_params):
    if is_deep(solver_params):
        raise NotImplementedError

    eps = 1e-32
    pred = clf.predict_proba(Sx)
    alpha_y = lambda y: (1 - rho[1 - y]) / (1 - rho[0] - rho[1])
    den = lambda p, y: rho[1 - y] * np.clip(p[1 - y], eps, 1 - eps) + (1 - rho[y]) * np.clip(p[y], eps, 1 - eps)
    return lx * np.array([alpha_y(y) - ((1 - rho[y]) * p[y]) / den(p, y) for p, y in zip(pred, Sy_noise)])


def robust_margin_selection(clf, Sx, Sy_noise, rho, lx, q, solver_params):
    pred = [p[y] for p, y in zip(clf.predict_proba(Sx), Sy_noise)]
    K = get_noise_matrix(rho)
    num = np.array([K[0, y] for y in Sy_noise])
    den = np.array([np.sum(K[:, y]) for y in Sy_noise])
    den[den == 0] = 1.
    D = np.abs(pred - num / den)
    D = np.max(D) - D  # criterion has to be maximized
    return D
