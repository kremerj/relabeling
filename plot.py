import numpy as np
import warnings
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from matplotlib import cm
from distutils import dir_util
from argparse import ArgumentParser
from util import read_json


def plot_criterion():
    fs = 15
    rho = np.arange(0, 0.5, 0.001)
    sigma = np.arange(0, 1., 0.001)[1:]

    R, S = np.meshgrid(rho, sigma)
    C = (R * (1 - S) * (1 - R) * S) / (R * (1 - S) + (1 - R) * S) ** 2.

    fig, ax = plt.subplots()
    ax.set_xlabel(r"output $\sigma(\tilde{y}x)$", fontsize=fs)
    ax.set_ylabel(r"noise rate $\rho_{+1}$", fontsize=fs)
    cax = ax.contourf(S, R, C, cmap=cm.coolwarm)
    cbar = fig.colorbar(cax)
    cbar.set_label(r"selection criterion $s_{\mathrm{ML}}(x,\tilde{y})$", fontsize=fs)
    cbar.set_ticks([0.0, np.max(C)])
    cbar.set_ticklabels(['0', ''])
    cbar.ax.tick_params(labelsize=fs)
    ax.tick_params(labelsize=12)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 0.5))
    fig.savefig('output/ml_selection.pdf', bbox_inches='tight', pad_inches=0)


def plot_experiment(dataset, experiment, algorithm, ci, legend):
    experiment_params = read_json('config/experiment.json')
    dataset_params = read_json('config/dataset.json')
    plot_params = read_json('config/plot.json')

    if experiment == '':
        experiments = experiment_params.keys()
    else:
        experiments = [experiment]

    if dataset == '':
        datasets = dataset_params.keys()
    else:
        datasets = [dataset]

    algs = [k for k, v in plot_params['algs'].items() if v == 1] if algorithm == '' else [algorithm]
    ub_alg = 'robust ub weighted uncertainty relabeling'

    for experiment in experiments:
        exp_algs = algs + [ub_alg] if experiment == 'unbiased' else algs
        trials = range(experiment_params[experiment]['dataset']['n_repeat'])
        for dataset in datasets:
            if (dataset == 'baidu') and (experiment != 'deep') or (dataset != 'baidu') and (experiment == 'deep'):
                continue
            print('plotting experiment %s on dataset %s' % (experiment, dataset))
            results = dict()
            for alg in exp_algs:
                alg_func = alg.replace(' ', '_')
                fnames = ['output/experiment/%s/%s/%s/%02d.npy' % (experiment, dataset, alg_func, t) for t in trials]
                results[alg] = np.array([np.load(f) for f in fnames])

            color_palette = [plot_params['algs_to_colors'][k] for k in results.keys()]
            rv = np.dstack((v[:, :plot_params['style']['range']] for v in results.values()))
            rk = [plot_params['algs_to_labels'][k] for k in results.keys()]

            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            fig.canvas.set_window_title('relabeling')
            ax.tick_params(labelsize=plot_params['style']['ticksize'])

            xlabel = 'examples corrected in batches of size 64' if experiment == 'deep' else 'examples corrected'
            plt.ylabel('accuracy on test set', fontsize=plot_params['style']['fontsize'])
            plt.xlabel(xlabel, fontsize=plot_params['style']['fontsize'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                sns.tsplot(rv, condition=rk, legend=legend, color=color_palette, ci=ci)
            plt.ylim(ymin=np.min(results['noisy baseline']) if algorithm == '' else 0.5)

            if legend:
                plt.legend(frameon=True, loc='lower right', fontsize=plot_params['style']['fontsize'], framealpha=0.5)

            sns.despine()
            output_path = 'output/experiment/%s/figures' % experiment
            dir_util.mkpath(output_path)
            fig.savefig('%s/%s.pdf' % (output_path, dataset), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    parser = ArgumentParser(description="Plot")
    parser.add_argument('--figure', type=str, default='experiment', nargs='?')
    parser.add_argument('--dataset', type=str, default='', nargs='?')
    parser.add_argument('--experiment', type=str, default='', nargs='?')
    parser.add_argument('--algorithm', type=str, default='', nargs='?')
    parser.add_argument('--ci', type=int, default=99, nargs='?')
    parser.add_argument('--legend', type=int, default=1, nargs='?')
    args = parser.parse_args()
    dataset = args.dataset
    experiment = args.experiment
    algorithm = args.algorithm
    ci = args.ci
    legend = bool(args.legend)
    plot_figure = args.figure

    sns.set_style("white")
    sns.set_context("paper")
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    if plot_figure == 'experiment':
        plot_experiment(dataset, experiment, algorithm, ci, legend)
    else:
        plot_criterion()
