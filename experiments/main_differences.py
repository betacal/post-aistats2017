# Usage:
# Parallelized in multiple threads:
#   python -m scoop -n 4 main.py # where -n is the number of workers (
# threads)
# Not parallelized (easier to debug):
#   python main.py

from __future__ import division
import os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import calib.models.adaboost as our
import sklearn.ensemble as their
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from calib.utils.plots import plot_score_differences
from calib.utils.plots import plot_score_distributions


# Parallelization
import itertools
from scoop import futures

# Our classes and modules
from calib.utils.calibration import cv_calibration_map_differences
from calib.utils.dataframe import MyDataFrame
from calib.utils.functions import get_sets

# Our datasets module
from data_wrappers.datasets import Data
from data_wrappers.datasets import datasets_li2014
from data_wrappers.datasets import datasets_hempstalk2008
from data_wrappers.datasets import datasets_others
from data_wrappers.datasets import datasets_big


classifiers = {
                  'nbayes': GaussianNB(),
                  'logistic': LogisticRegression(),
                  'adao': our.AdaBoostClassifier(n_estimators=200),
                  'adas': their.AdaBoostClassifier(n_estimators=200),
                  'forest': RandomForestClassifier(n_estimators=200),
                  'mlp': MLPClassifier(),
                  'svm': SVC()
}
score_types = {
                  'nbayes': 'predict_proba',
                  'logistic': 'predict_proba',
                  'adao': 'predict_proba',
                  'adas': 'predict_proba',
                  'forest': 'predict_proba',
                  'mlp': 'predict_proba',
                  'svm': 'sigmoid'
}

seed_num = 42
mc_iterations = 1
n_folds = 5
plot_differences = False
classifier_name = 'mlp'
if plot_differences:
    results_path = 'results_differences/' + classifier_name
else:
    results_path = 'results_distributions/' + classifier_name
classifier = classifiers[classifier_name]
score_type = score_types[classifier_name]

columns = ['a_dist', 'a_trained', 'b_dist', 'b_trained', 'm_dist', 'm_trained']


def compute_all(args):
    (name, dataset, n_folds, mc) = args
    np.random.seed(mc)
    skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                          shuffle=True)
    df = MyDataFrame(columns=columns)
    test_folds = skf.test_folds
    class_counts = np.bincount(dataset.target)
    if np.alen(class_counts) > 2:
        majority = np.argmax(class_counts)
        t = np.zeros_like(dataset.target)
        t[dataset.target == majority] = 1
    else:
        t = dataset.target
    fold_range = [0]
    if plot_differences:
        fold_range = np.arange(n_folds)
    for test_fold in fold_range:
        x_train, y_train, x_test, y_test = get_sets(dataset.data, t, test_fold,
                                                    test_folds)
        a, b, m, df_pos, df_neg, ccv = cv_calibration_map_differences(classifier,
                                                                      x_train,
                                                                      y_train,
                                                                      cv=3,
                                                                      score_type=score_type)
        rows = np.hstack([a[:, 0].reshape(-1, 1), a[:, 1].reshape(-1, 1),
                          b[:, 0].reshape(-1, 1), b[:, 1].reshape(-1, 1),
                          m[:, 0].reshape(-1, 1), m[:, 1].reshape(-1, 1)])
        df = df.append_rows(rows)

        if not plot_differences:
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            fig_distributions = plot_score_distributions(df_pos, df_neg,
                                                         ccv.calibrator)

            fig_distributions.savefig(os.path.join(results_path, name + ".pdf"))
    return df


if __name__ == '__main__':
    dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
                             datasets_others) - set(['lung-cancer', 'glass']))
    # dataset_names = datasets_big
    # dataset_names = ['glass']
    dataset_names.sort()
    df_all = MyDataFrame(columns=columns)

    data = Data(dataset_names=dataset_names)

    for name, dataset in data.datasets.iteritems():
        df = MyDataFrame(columns=columns)
        print(dataset)

        mcs = np.arange(mc_iterations)
        # All the arguments as a list of lists
        args = [[name], [dataset], [n_folds], mcs]
        args = list(itertools.product(*args))

        # if called with -m scoop
        if '__loader__' in globals():
            dfs = futures.map(compute_all, args)
        else:
            dfs = map(compute_all, args)

        df = df.concat(dfs)
        df_all = df_all.append(df)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if plot_differences:
        fig_differences = plot_score_differences(df_all['a_dist'].values,
                                                 df_all['a_trained'].values,
                                                 'a', limits=[0, 5])

        fig_differences.savefig(os.path.join(results_path,
                                             classifier_name + "_parameter_a.pdf"))

        fig_differences = plot_score_differences(df_all['b_dist'].values,
                                                 df_all['b_trained'].values,
                                                 'b', limits=[0, 5])

        fig_differences.savefig(os.path.join(results_path,
                                             classifier_name + "_parameter_b.pdf"))

        fig_differences = plot_score_differences(df_all['m_dist'].values,
                                                 df_all['m_trained'].values, 'm')

        fig_differences.savefig(os.path.join(results_path,
                                             classifier_name + "_parameter_m.pdf"))
