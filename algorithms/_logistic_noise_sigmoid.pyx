#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

from libc.math cimport log, exp

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t


cdef DTYPE_t _inner_noise_log_logistic_sigmoid(DTYPE_t x, DTYPE_t rho_neg, DTYPE_t rho_pos):
    """Log of the logistic sigmoid function log(1 / (1 + e ** -x))"""
    if x > 0:
        return -log(1 + rho_neg * exp(-x) - rho_pos) if rho_pos < 1 else x - log(rho_neg)
    else:
        return x - log(rho_neg + (1-rho_pos) * exp(x)) if rho_neg > 0 else -log(1 - rho_pos)


def _log_logistic_noise_sigmoid(int n_samples, int n_features, 
                           np.ndarray[DTYPE_t, ndim=2] X,
                           np.ndarray[DTYPE_t, ndim=2] out,
                           np.ndarray[DTYPE_t, ndim=1] rho_neg,
                           np.ndarray[DTYPE_t, ndim=1] rho_pos):
    for i in range(n_samples):
        for j in range(n_features):
            out[i, j] = _inner_noise_log_logistic_sigmoid(X[i, j], rho_neg[i], rho_pos[i])
    return out