import numpy as np
import json
from sklearn.linear_model import LogisticRegression


def read_json(filename):
    with open(filename, 'r') as f:
        dictionary = json.load(f)
    return dictionary


def init_logistic(solver_params, rb_alg, rho):
    from algorithms.unbiased_logistic import UnbiasedLogisticRegression
    from algorithms.ml_noise_logistic import MLNoiseLogisticRegression

    if rb_alg == 'robust_ub':
        clf = UnbiasedLogisticRegression(rho=rho, **solver_params)
    elif rb_alg == 'robust_ml':
        clf = MLNoiseLogisticRegression(rho=rho, **solver_params)
    elif rb_alg in ['robust_em', 'robust_map']:
        clf = MLNoiseLogisticRegression(rho=rho, optimize_rho=True, **solver_params)
    else:
        clf = LogisticRegression(**solver_params)

    return clf


def init_deep(solver_params, rb_alg, rho):
    from algorithms.convnet import ConvNet
    
    return ConvNet(solver_params['model'], solver_params['train_batch_size'], solver_params['test_batch_size'],
                   robust=rb_alg, rho=rho)


def is_deep(solver_params):
    return 'model' in solver_params


def get_classifier(solver_params, rb_alg=None, rho=np.zeros(2)):
    if is_deep(solver_params):
        clf = init_deep(solver_params, rb_alg, rho)
        batch_size = solver_params['train_batch_size']
    else:
        clf = init_logistic(solver_params, rb_alg, rho)
        batch_size = 1
    return clf, batch_size
