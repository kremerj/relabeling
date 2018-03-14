import numpy as np
import scipy.sparse as sp
from scipy import optimize

from sklearn.utils.testing import assert_array_almost_equal

from sklearn.datasets import load_iris, make_classification

from sklearn.linear_model.logistic import _logistic_loss as std_logistic_loss
from algorithms.unbiased_logistic import _logistic_loss, _logistic_loss_and_grad
from algorithms.ml_noise_logistic import _logistic_loss as ml_noise_logistic_loss


def generate_matrix_q(noise_level):
    q = np.zeros((2,2))
    q[0,0] = 1 - noise_level[0]
    q[0,1] = noise_level[1]
    q[1,0] = noise_level[0]
    q[1,1] = 1 - noise_level[1]

    return q


def test_logistic_loss_and_grad(N, q, rho, alpha):
    X_ref, y = make_classification(n_samples=N)
    y[y==0] = -1
    n_features = X_ref.shape[1]

    X_sp = X_ref.copy()
    X_sp[X_sp < .1] = 0
    X_sp = sp.csr_matrix(X_sp)
    for X in (X_ref, X_sp):
        w = np.random.rand(n_features)
        #w = np.append(w,1)

        # # First check that our derivation of the grad is correct
        loss, grad = _logistic_loss_and_grad(w, X, y, alpha=alpha, q=q, rho=rho)
        loss_ = _logistic_loss(w, X, y, alpha=alpha, q=q, rho=rho)
        assert_array_almost_equal(loss, loss_, decimal=12)

        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=alpha, q=q, rho=rho)[0], 1e-8
            )
        assert_array_almost_equal(grad, approx_grad, decimal=4)

        # Second check that our intercept implementation is good
        w = np.append(w, [0])
        loss_interp, grad_interp = _logistic_loss_and_grad(
            w, X, y, alpha=alpha, q=q, rho=rho
            )
        assert_array_almost_equal(loss, loss_interp)

        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=alpha, q=q, rho=rho)[0], 1e-8
            )
        assert_array_almost_equal(grad_interp, approx_grad, decimal=4)
    return True


N = 1000
rho = [0.3,0.2]
alpha = 1.0
q = np.random.randint(2, size=N)
print('test gradient')
if test_logistic_loss_and_grad(N=N, q=q, rho=rho, alpha=alpha):
    print('test passed')

np.random.seed(401)
w = np.random.rand(4)
X = np.random.rand(10,4)
y = np.random.choice([-1,1], size=10)
q = np.random.choice(2, size=10)
print(y)
y0 = np.array(y == 1, dtype=int)
alpha = 1
rho = [0.2,0.1]

# standard loss with regularization
print('std loss')
print(np.sum(np.log(1+np.exp(-y*np.dot(X,w)))) + 0.5 * alpha * np.dot(w,w))
print(_logistic_loss(w, X, y, alpha, rho=[0., 0.]))
print(std_logistic_loss(w, X, y, alpha))
print(ml_noise_logistic_loss(w, X, y, alpha, rho=[0., 0.]))

# unbiased loss with regularization
print('unbiased loss')
print(np.sum(np.log(1+np.exp(-y*np.dot(X,w)))*q) + np.sum((1-q) * ((1-np.take(rho, 1-y0)) * np.log(1+np.exp(-y*np.dot(X,w))) - np.take(rho, y0) * np.log(1+np.exp(y*np.dot(X,w))))) / (1-rho[0]-rho[1]) + 0.5 * alpha * np.dot(w, w))
print(_logistic_loss(w, X, y, alpha, rho=rho, q=q))