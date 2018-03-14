import numpy as np
import itertools
from argparse import ArgumentParser
from joblib import Parallel, delayed
from distutils import dir_util
from util import read_json
from dataset import generate_dataset

from relabeling import (noisy_baseline,
                        clean_baseline,
                        passive_relabeling,
                        uncertainty_sampling,
                        uncertainty_relabeling,
                        weighted_uncertainty_relabeling,
                        robust_uncertainty_relabeling,
                        robust_ml_uncertainty_relabeling,
                        robust_em_uncertainty_relabeling,
                        robust_map_uncertainty_relabeling,
                        robust_ub_weighted_uncertainty_relabeling,
                        robust_ml_weighted_uncertainty_relabeling,
                        robust_em_weighted_uncertainty_relabeling,
                        robust_map_weighted_uncertainty_relabeling)


def run_experiment(r, alg, dataset, experiment, params, solver_params):
    print('%s, trial: %i, dataset %s, experiment %s' % (alg, r, dataset, experiment))

    # sample a new dataset
    Sx, Sy_clean, Sy_noise, Tx, Ty = generate_dataset(r, dataset, params)

    # perform learning
    alg_func_name = alg.replace(' ', '_')
    func = eval(alg_func_name)
    params['random_state'] = r
    scores = func(Sx, Sy_clean, Sy_noise, Tx, Ty, params, solver_params)
    output_path = 'output/experiment/%s/%s/%s' % (experiment, dataset, alg_func_name)
    dir_util.mkpath(output_path)
    np.save('%s/%02d.npy' % (output_path, r), scores)


if __name__ == '__main__':
    parser = ArgumentParser(description="Relabeling")
    parser.add_argument('--experiment', type=str, default='', nargs='?')
    parser.add_argument('--algorithm', type=str, default='', nargs='?')
    parser.add_argument('--dataset', type=str, default='', nargs='?')
    parser.add_argument('--trial', type=int, default=-1, nargs='?')
    parser.add_argument('--jobs', type=int, default=1, nargs='?')
    args = parser.parse_args()
    experiment = args.experiment
    algorithm = args.algorithm
    dataset = args.dataset
    trial = args.trial
    n_jobs = args.jobs

    experiment_params = read_json('config/experiment.json')
    dataset_params = read_json('config/dataset.json')
    solver_params = read_json('config/solver.json')

    if algorithm == '':
        relabeling_params = read_json('config/relabeling.json')
        ub_alg = 'robust ub weighted uncertainty relabeling'  # not implemented for deep network
        algs = [k for k, v in relabeling_params['algs'].items() if
                v == 1 and not (experiment == 'deep' and k == ub_alg)]
    else:
        algs = [algorithm]

    # add parameters to specific dataset and solver parameters
    solver_params.update(experiment_params[experiment]['solver'])
    dataset_params[dataset].update(experiment_params[experiment]['dataset'])

    results = dict.fromkeys(algs)
    params = dataset_params[dataset]
    parallel = Parallel(n_jobs=n_jobs, verbose=50)
    R = range(params['n_repeat']) if trial == -1 else [trial]
    parallel(delayed(run_experiment)(r, alg, dataset, experiment, params, solver_params) for alg, r in
             itertools.product(algs, R))
