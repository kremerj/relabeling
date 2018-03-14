import numpy as np


def update_confusion(pred, labels, K, q, beta=np.ones((2, 2)), eps=1e-32):
    if np.any(beta[0, :] == 0):
        beta[0, :] = [1., 1.]

    if np.any(beta[1, :] == 0):
        beta[1, :] = [1., 1.]

    qnoise = np.array(q == 0, dtype=np.bool)
    yb = np.array(labels > 0, dtype=int)[qnoise]
    gamma = K
    z = pred

    S0 = gamma[0, 0] * (1 - z) + gamma[1, 0] * z
    S1 = gamma[0, 1] * (1 - z) + gamma[1, 1] * z

    div0 = np.nan_to_num(1. / S0)
    div1 = np.nan_to_num(1. / S1)

    a0 = gamma[0, 0] * np.sum((1 - yb) * div0 * (1 - z))
    b0 = gamma[0, 1] * np.sum(yb * div1 * (1 - z))
    s0 = a0 + b0 + beta[0, 0] + beta[0, 1] - 2

    gamma[0, 0] = np.clip((a0 + beta[0, 0] - 1) / s0 if s0 > 0 else 1., eps, 1. - eps)
    gamma[0, 1] = 1 - gamma[0, 0]

    a1 = gamma[1, 1] * np.sum(yb * div1 * z)
    b1 = gamma[1, 0] * np.sum((1 - yb) * div0 * z)
    s1 = a1 + b1 + beta[1, 0] + beta[1, 1] - 2

    gamma[1, 1] = np.clip((a1 + beta[1, 1] - 1) / s1 if s1 > 0 else 1., eps, 1. - eps)
    gamma[1, 0] = 1 - gamma[1, 1]

    return gamma


def update_noise_rates(pred, labels, rho, q, beta=np.ones((2, 2))):
    gamma = get_noise_matrix(rho)
    gamma = update_confusion(pred, labels, gamma, q, beta)
    rho = [gamma[0, 1], gamma[1, 0]]
    return rho


def get_noise_matrix(rho):
    return np.array([[1 - rho[0], rho[0]], [rho[1], 1 - rho[1]]])


def estimate_noise_rates(weights, yn, yc, rb_alg=None):
    # estimate unnormalized joint probabilities p(y_tilde=0, y=1), p(y_tilde=1, y=0)
    pjoint01 = np.sum(weights * (yn == 0) * (yc == 1))
    pjoint10 = np.sum(weights * (yn == 1) * (yc == 0))

    # estimate unnormalized marginal probabilities p(y=0), p(y=1)
    pc1 = np.sum(weights * (yc == 1))
    pc0 = np.sum(weights * (yc == 0))

    # estimate noise rates
    rho_est_minus = pjoint10 / pc0 if pc0 > 0 else pjoint10 / len(yn)
    rho_est_plus = pjoint01 / pc1 if pc1 > 0 else pjoint01 / len(yn)

    # avoid pathological cases and force random sampling
    # rho_plus = 0 and rho_minus = 1 and vice versa
    if (rho_est_minus == 1) and (rho_est_plus == 0) or (rho_est_minus == 0) and (rho_est_plus == 1):
        return 0., 0.

    if rb_alg == 'robust_ub' and rho_est_plus + rho_est_minus >= 1.:
        return min(rho_est_minus, 0.45), min(rho_est_plus, 0.45)

    return rho_est_minus, rho_est_plus
