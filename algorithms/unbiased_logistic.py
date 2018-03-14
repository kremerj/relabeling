import numbers
import warnings

import numpy as np
from scipy import optimize
from scipy.special import expit
import scipy.sparse as sp

from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.linear_model.sag import sag_solver
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.svm.base import _fit_liblinear
from sklearn.utils import check_array, check_consistent_length, compute_class_weight
from sklearn.utils import check_random_state
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.utils.optimize import newton_cg
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.externals.joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _check_solver_option, _intercept_dot


def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None, copy=False,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None, rho=None, q=None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,)
        Input data, target values.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag'}
        Numerical solver to use.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    copy : bool, default False
        Whether or not to produce a copy of the data. A copy is not required
        anymore. This parameter is deprecated and will be removed in 0.19.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    intercept_scaling : float, default 1.
        This parameter is useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial'}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Works only for the 'lbfgs' and
        'newton-cg' solvers.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slighly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.
    """
    if copy:
        warnings.warn("A copy is not required anymore. The 'copy' parameter "
                      "is deprecated and will be removed in 0.19.",
                      DeprecationWarning)

    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    _check_solver_option(solver, multi_class, penalty, dual)

    # Preprocessing.
    if check_input or copy:
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        y = check_array(y, ensure_2d=False, copy=copy, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape
    classes = np.unique(y)
    random_state = check_random_state(random_state)

    if pos_class is None and multi_class != 'multinomial':
        if (classes.size > 2):
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    if sample_weight is not None:
        sample_weight = np.array(sample_weight, dtype=np.float64, order='C')
        check_consistent_length(y, sample_weight)
    else:
        sample_weight = np.ones(X.shape[0])

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()

    if isinstance(class_weight, dict) or multi_class == 'multinomial':
        if solver == "liblinear":
            if classes.size == 2:
                # Reconstruct the weights with keys 1 and -1
                temp = {1: class_weight[pos_class],
                        -1: class_weight[classes[0]]}
                class_weight = temp.copy()
            else:
                raise ValueError("In LogisticRegressionCV the liblinear "
                                 "solver cannot handle multiclass with "
                                 "class_weight of type dict. Use the lbfgs, "
                                 "newton-cg or sag solvers or set "
                                 "class_weight='balanced'")
        else:
            class_weight_ = compute_class_weight(class_weight, classes, y)
            sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == 'ovr':
        w0 = np.zeros(n_features + int(fit_intercept))
        mask_classes = np.array([-1, 1])
        mask = (y == pos_class)
        y_bin = np.ones(y.shape, dtype=np.float64)
        y_bin[~mask] = -1.
        # for compute_class_weight

        # 'auto' is deprecated and will be removed in 0.19
        if class_weight in ("auto", "balanced"):
            class_weight_ = compute_class_weight(class_weight, mask_classes,
                                                 y_bin)
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        lbin = LabelBinarizer()
        Y_binarized = lbin.fit_transform(y)
        if Y_binarized.shape[1] == 1:
            Y_binarized = np.hstack([1 - Y_binarized, Y_binarized])
        w0 = np.zeros((Y_binarized.shape[1], n_features + int(fit_intercept)),
                      order='F')

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    'Initialization coef is of shape %d, expected shape '
                    '%d or %d' % (coef.size, n_features, w0.size))
            w0[:coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_vectors = classes.size
            if n_vectors == 2:
                n_vectors = 1

            if (coef.shape[0] != n_vectors or
                    coef.shape[1] not in (n_features, n_features + 1)):
                raise ValueError(
                    'Initialization coef is of shape (%d, %d), expected '
                    'shape (%d, %d) or (%d, %d)' % (
                        coef.shape[0], coef.shape[1], classes.size,
                        n_features, classes.size, n_features + 1))
            w0[:, :coef.shape[1]] = coef

    if multi_class == 'multinomial':
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        w0 = w0.ravel()
        target = Y_binarized
        if solver == 'lbfgs':
            func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
        elif solver == 'newton-cg':
            func = lambda x, *args: _multinomial_loss(x, *args)[0]
            grad = lambda x, *args: _multinomial_loss_grad(x, *args)[1]
            hess = _multinomial_grad_hess
    else:
        target = y_bin
        if solver == 'lbfgs':
            func = lambda *args: _logistic_loss_and_grad(rho=rho, q=q, *args)
        elif solver == 'newton-cg':
            func = _logistic_loss
            grad = lambda x, *args: _logistic_loss_and_grad(x, *args)[1]
            hess = _logistic_grad_hess

    coefs = list()
    warm_start_sag = {'coef': w0}
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            try:
                w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, target, 1. / C, sample_weight),
                    iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)
            except TypeError:
                # old scipy doesn't have maxiter
                w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, target, 1. / C, sample_weight),
                    iprint=(verbose > 0) - 1, pgtol=tol)
            if info["warnflag"] == 1 and verbose > 0:
                warnings.warn("lbfgs failed to converge. Increase the number "
                              "of iterations.")
            try:
                n_iter_i = info['nit'] - 1
            except:
                n_iter_i = info['funcalls'] - 1
        elif solver == 'newton-cg':
            args = (X, target, 1. / C, sample_weight)
            w0, n_iter_i = newton_cg(hess, func, grad, w0, args=args,
                                     maxiter=max_iter, tol=tol)
        elif solver == 'liblinear':
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X, target, C, fit_intercept, intercept_scaling, class_weight,
                penalty, dual, verbose, max_iter, tol, random_state)
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver == 'sag':
            w0, n_iter_i, warm_start_sag = sag_solver(
                X, target, sample_weight, 'log', 1. / C, max_iter, tol,
                verbose, random_state, False, max_squared_sum,
                warm_start_sag)
        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                             "'newton-cg', 'sag'}, got '%s' instead" % solver)

        if multi_class == 'multinomial':
            multi_w0 = np.reshape(w0, (classes.size, -1))
            if classes.size == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0)
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return coefs, np.array(Cs), n_iter

def _logistic_loss(w, X, y, alpha, sample_weight=None, rho=None, q=None):
    """Computes the logistic loss.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : ndarray, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    out : float
        Logistic loss.
    """
    # 0: noise, 1: clean
    if q is None:
        q = np.zeros_like(y)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    y01 = np.array(y == 1, dtype=int)
    w, c, yz = _intercept_dot(w, X, y)

    loss_yzp = -log_logistic(+yz)
    loss_yzn = -log_logistic(-yz)
    wp = (1-np.take(rho, 1-y01)) / (1-rho[0]-rho[1])
    wn = ( -np.take(rho,   y01)) / (1-rho[0]-rho[1])
    noise_loss = np.sum(sample_weight * (1-q) * (wp * loss_yzp + wn * loss_yzn))
    clean_loss = np.sum(sample_weight * q * loss_yzp)
    out = clean_loss + noise_loss + .5 * alpha * np.dot(w, w)

    return out

def _logistic_loss_and_grad(w, X, y, alpha, sample_weight=None, rho=None, q=None):
    """Computes the logistic loss and gradient.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : ndarray, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    out : float
        Logistic loss.

    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """

    _, n_features = X.shape
    grad = np.empty_like(w)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    # 0: noise, 1: clean
    if q is None:
        q = np.zeros_like(y)

    y01 = np.array(y == 1, dtype=int)
    w, c, yz = _intercept_dot(w, X, y)

    loss_yzp = -log_logistic(+yz)
    loss_yzn = -log_logistic(-yz)
    wp = 1 - np.take(rho, 1 - y01)
    wn = np.take(rho, y01)
    noise_loss = np.sum(sample_weight * (1-q) * (wp * loss_yzp - wn * loss_yzn)) / (1 - rho[0] - rho[1])
    clean_loss = np.sum(sample_weight * q * loss_yzp)
    out = clean_loss + noise_loss + .5 * alpha * np.dot(w, w)

    z = expit(yz)
    z0 = sample_weight * (q * (z-1) * y + (1-q) * (wp * (z-1) * y + wn * z * y))
    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()

    return out, grad


class UnbiasedLogisticRegression(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1, rho=None):

        self.rho = rho
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None, q=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64, 
                         order="C")
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        _check_solver_option(self.solver, self.multi_class, self.penalty,
                             self.dual)

        if self.solver == 'liblinear':
            self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
                X, y, self.C, self.fit_intercept, self.intercept_scaling,
                self.class_weight, self.penalty, self.dual, self.verbose,
                self.max_iter, self.tol, self.random_state)
            self.n_iter_ = np.array([n_iter_])
            return self

        max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef,
                                        self.intercept_[:, np.newaxis],
                                        axis=1)

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        # Hack so that we iterate only once for the multinomial case.
        if self.multi_class == 'multinomial':
            classes_ = [None]
            warm_start_coef = [warm_start_coef]

        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        backend = 'threading' if self.solver == 'sag' else 'multiprocessing'
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               backend=backend)(
            path_func(X, y, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept, tol=self.tol,
                      verbose=self.verbose, solver=self.solver, copy=False,
                      multi_class=self.multi_class, max_iter=self.max_iter,
                      class_weight=self.class_weight, check_input=False,
                      random_state=self.random_state, coef=warm_start_coef_,
                      max_squared_sum=max_squared_sum,
                      sample_weight=sample_weight, rho=self.rho, q=q)
            for (class_, warm_start_coef_) in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        if self.multi_class == 'multinomial':
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(n_classes, n_features +
                                            int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self


def test_logistic_loss_and_grad():
    X_ref, y = make_classification(n_samples=20)
    n_features = X_ref.shape[1]

    rho = [0.0, 0.5]
    np.random.seed(0)
    q = np.random.randint(0, 2, len(y))

    X_sp = X_ref.copy()
    X_sp[X_sp < .1] = 0
    X_sp = sp.csr_matrix(X_sp)
    for X in (X_ref, X_sp):
        w = np.random.rand(n_features)

        # First check that our derivation of the grad is correct
        loss, grad = _logistic_loss_and_grad(w, X, y, alpha=1., rho=rho, q=q)
        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=1., rho=rho, q=q)[0], 1e-3
            )
        assert_array_almost_equal(grad, approx_grad, decimal=2)

        # Second check that our intercept implementation is good
        w = np.random.rand(n_features + 1)
        loss_interp, grad_interp = _logistic_loss_and_grad(
            w, X, y, alpha=1., rho=rho, q=q
            )
        assert_array_almost_equal(loss, loss_interp)

        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=1., rho=rho, q=q)[0], 1e-3
            )
        assert_array_almost_equal(grad_interp, approx_grad, decimal=2)


if __name__ == '__main__':
    test_logistic_loss_and_grad()