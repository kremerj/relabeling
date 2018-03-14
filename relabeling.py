import numpy as np
from sklearn.utils import check_random_state
from selection import ub_selection, ml_selection, robust_margin_selection, margin_selection, agnostic_selection
from noise import estimate_noise_rates
from util import get_classifier, is_deep


def robust_ub_weighted_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'robust_ub',
                                          ub_selection)


def robust_ml_weighted_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'robust_ml',
                                          ml_selection)


def robust_em_weighted_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'robust_em',
                                          ml_selection)


def robust_map_weighted_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'robust_map',
                                          ml_selection)


def robust_ml_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'robust_ml',
                                          robust_margin_selection)


def robust_em_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'robust_em',
                                          robust_margin_selection)


def robust_map_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'robust_map',
                                          robust_margin_selection)


def robust_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'non_robust',
                                          robust_margin_selection)


def weighted_uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'non_robust',
                                          agnostic_selection)


def importance_weighted_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, rb_alg, selection):
    n_burnin = params['n_burnin'] if rb_alg in ['robust_ml', 'robust_ub'] else 0
    rng = check_random_state(params['random_state'])
    n_train = len(Sy_clean)
    scores = np.zeros(n_train)
    relabel_indices = -np.ones(n_train, dtype=int)

    # fit initially to get an estimate of the classifier's uncertainty
    clf, batch_size = get_classifier(solver_params)

    # compute sample magnitudes
    lx = None if is_deep(solver_params) else np.linalg.norm(Sx, axis=1)

    # indicator for clean/noise examples 0: noise, 1: clean
    q = np.zeros(n_train)

    # probability distribution for drawing examples
    prob = np.zeros(n_train)

    # sample weights
    weights = np.zeros(n_train)

    # initial estimate of the noise rate / prior probabilities
    rho = rng.uniform(0.0, 0.4, 2) if rb_alg in ['robust_em', 'robust_map'] else [0., 0.]

    # indicates that we should refit our classifier
    refit = True

    b = 0
    t = 1
    Sy_noise_orig = Sy_noise.copy()

    while b < params['n_sample']:
        if b < n_burnin:
            prob[:] = 1. / n_train

            if refit:
                clf.fit(Sx, Sy_noise)
                refit = False
        else:
            if b == n_burnin:
                if rb_alg.startswith('robust') and refit:
                    clf, _ = get_classifier(solver_params, rb_alg, rho)
            elif rb_alg not in ['robust_em', 'robust_map']:
                clf.rho = rho

            if refit:
                if rb_alg == 'robust_map':
                    prior = estimate_noise_rates(weights, Sy_noise_orig, Sy_clean, rb_alg)
                    beta = b * np.array([[1 - prior[0], prior[0]], [prior[1], 1 - prior[1]]])
                    clf.fit(Sx, Sy_noise, q=q, beta=beta)
                elif rb_alg.startswith('robust'):
                    clf.fit(Sx, Sy_noise, q=q)
                else:
                    clf.fit(Sx, Sy_noise)

                if rb_alg in ['robust_em', 'robust_map']:
                    rho = clf.rho

                # expected gradient length
                df = selection(clf, Sx, Sy_noise, rho, lx, q, solver_params)
                sum_df = np.sum(df)
                refit = False

            # normalize criterion to probability distribution
            pmin = 1. / (n_train * t ** params['kappa'])
            prob[:] = df / sum_df if sum_df > 0 else 1. / n_train
            prob = pmin + (1 - n_train * pmin) * prob

        relabel_index = rng.choice(n_train, p=prob)
        weights[relabel_index] += 1. / prob[relabel_index]

        if relabel_index not in relabel_indices[:b]:
            relabel_indices[b] = relabel_index
            rho = estimate_noise_rates(weights, Sy_noise_orig, Sy_clean, rb_alg)
            Sy_noise[relabel_index] = Sy_clean[relabel_index]
            q[relabel_index] = 1
            b += 1

            if b % batch_size == 0:
                refit = True
                scores[b - batch_size: b] = clf.score(Tx, Ty)
        t += 1

    return scores


def uncertainty_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return greedy_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'relabeling')


def uncertainty_sampling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return greedy_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'sampling')


def passive_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return greedy_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, 'passive')


def greedy_relabeling(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params, alg):
    n_train = len(Sy_clean)
    scores = np.zeros(n_train)
    rng = check_random_state(params['random_state'])
    relabel_indices = -np.ones(n_train, dtype=int)

    # fit initially to get an estimate of the classifier's uncertainty
    clf, batch_size = get_classifier(solver_params)
    clf.fit(Sx, Sy_noise)

    for b in range(0, params['n_sample'], batch_size):
        # absolute distance from decision function
        df = margin_selection(clf, Sx)

        # exclude already relabeled examples
        cand = np.delete(range(params['n_sample']), relabel_indices[:b])
        df[relabel_indices[:b]] = np.finfo(float).min

        # iteratively query for more clean labels
        if alg == 'relabeling' or alg == 'sampling' and len(np.unique(Sy_noise[relabel_indices[:b]])) == 2:
            relabel = np.argsort(df)[:-1 - batch_size:-1]
        else:
            relabel = rng.choice(cand, size=batch_size, replace=False)

        relabel_indices[b: b + batch_size] = relabel
        Sy_noise[relabel] = Sy_clean[relabel]

        if alg == 'sampling':
            subsample = relabel_indices[:b + batch_size]
            if len(np.unique(Sy_noise[subsample])) == 2:
                clf.fit(Sx[subsample, :], Sy_noise[subsample])
        elif alg in ['relabeling', 'passive']:
            clf.fit(Sx, Sy_noise)

        scores[b: b + batch_size] = clf.score(Tx, Ty)

    return scores


def noisy_baseline(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return baseline(Sx, Sy_noise, Tx, Ty, params, solver_params)


def clean_baseline(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params):
    return baseline(Sx, Sy_clean, Tx, Ty, params, solver_params)


def baseline(Sx, Sy, Tx, Ty, params, solver_params):
    n_train = len(Sy)
    clf, _ = get_classifier(solver_params)
    clf.fit(Sx, Sy)
    scores = clf.score(Tx, Ty) * np.ones(n_train)
    return scores
