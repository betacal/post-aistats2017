from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import beta

from calib.utils.functions import fit_beta_moments


def reliability_diagram(prob, Y, marker='--', label='', alpha=1, linewidth=1,
                        ax_reliability=None, clip=True):
    '''
        alpha= Laplace correction, default add-one smoothing
    '''
    bins = np.linspace(0,1+1e-16,11)
    prob = np.clip(prob, 0, 1)
    hist_tot = np.histogram(prob, bins=bins)
    hist_pos = np.histogram(prob[Y == 1], bins=bins)
    # Compute the centroids of every bin
    centroids = [np.mean(np.append(
                 prob[np.where(np.logical_and(prob >= bins[i],
                                              prob < bins[i+1]))],
                 bins[i]+0.05)) for i in range(len(hist_tot[1])-1)]

    proportion = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+alpha*2)
    if ax_reliability is None:
        ax_reliability = plt.subplot(111)

    ax_reliability.plot(centroids, proportion, marker, linewidth=linewidth,
                        label=label)


def plot_reliability_diagram(scores_set, labels, legend_set,
                             original_first=False, alpha=1, **kwargs):
    fig_reliability = plt.figure('reliability_diagram')
    fig_reliability.clf()
    ax_reliability = plt.subplot(111)
    ax = ax_reliability
    # ax.set_title('Reliability diagram')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    n_lines = len(legend_set)
    if original_first:
        bins = np.linspace(0, 1, 11)
        hist_tot = np.histogram(scores_set[0], bins=bins)
        hist_pos = np.histogram(scores_set[0][labels == 1], bins=bins)
        edges = np.insert(bins, np.arange(len(bins)), bins)
        empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
        empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                                empirical_p)
        ax.plot(edges[1:-1], empirical_p, label='empirical')

    skip = original_first
    for (scores, legend) in zip(scores_set, legend_set):
        if skip and original_first:
            skip = False
        else:
            reliability_diagram(scores, labels, marker='x-',
                    label=legend, linewidth=n_lines, alpha=alpha, **kwargs)
            n_lines -= 1
    if original_first:
        ax.plot(scores_set[0], labels, 'kx', label=legend_set[0],
                markersize=9, markeredgewidth=1)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig_reliability


def plot_reliability_map(scores_set, prob, legend_set,
                         original_first=False, alpha=1, **kwargs):
    fig_reliability_map = plt.figure('reliability_map')
    fig_reliability_map.clf()
    ax_reliability_map = plt.subplot(111)
    ax = ax_reliability_map
    # ax.set_title('Reliability map')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    n_lines = len(legend_set)
    if original_first:
        bins = np.linspace(0, 1, 11)
        hist_tot = np.histogram(prob[0], bins=bins)
        hist_pos = np.histogram(prob[0][prob[1] == 1], bins=bins)
        edges = np.insert(bins, np.arange(len(bins)), bins)
        empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
        empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                                empirical_p)
        ax.plot(edges[1:-1], empirical_p, label='empirical')

    skip = original_first
    for (scores, legend) in zip(scores_set, legend_set):
        if skip and original_first:
            skip = False
        else:
            if legend == 'uncalib':
                ax.plot([np.nan], [np.nan], '-', linewidth=n_lines,
                        **kwargs)
            else:
                ax.plot(prob[2], scores, '-', label=legend, linewidth=n_lines,
                        **kwargs)
            n_lines -= 1
    if original_first:
        ax.plot(prob[0], prob[1], 'kx',
                label=legend_set[0], markersize=9, markeredgewidth=1)
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig_reliability_map


def plot_niculescu_mizil_map(scores_set, prob, legend_set, alpha=1, **kwargs):
    from matplotlib import rc
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    fig_reliability_map = plt.figure('reliability_map')
    fig_reliability_map.clf()
    ax_reliability_map = plt.subplot(111)
    ax = ax_reliability_map
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_xlabel((r'$s$'), fontsize=16)
    ax.set_ylabel((r'$\hat{p}$'), fontsize=16)
    n_lines = len(legend_set)
    bins = np.linspace(0, 1, 11)
    hist_tot = np.histogram(prob[0], bins=bins)
    hist_pos = np.histogram(prob[0][prob[1] == 1], bins=bins)
    centers = (bins[:-1] + bins[1:])/2.0
    empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
    ax.plot(centers, empirical_p, 'ko', label='empirical')

    for (scores, legend) in zip(scores_set, legend_set):
        if legend != 'uncalib':
            ax.plot(prob[2], scores, '-', label=legend, linewidth=n_lines,
                    **kwargs)
        n_lines -= 1
    ax.legend(loc='upper left')
    return fig_reliability_map


def remove_low_high(a1, a2, low, high):
    idx1 = np.logical_and(a1 > low, a1 <= high)
    idx2 = np.logical_and(a2 > low, a2 <= high)
    idx = np.logical_and(idx1, idx2)
    return a1[idx], a2[idx]


def plot_score_differences(dist, trained, parameter, limits=None):
    if limits is not None:
        dist, trained = remove_low_high(dist, trained, limits[0], limits[1])

    fig_score_differences = plt.figure('score_differences')
    fig_score_differences.clf()
    ax_score_differences = plt.subplot(111)
    ax = ax_score_differences
    ax.set_xlabel(parameter + '_dist', fontsize=16)
    ax.set_ylabel(parameter + '_trained', fontsize=16)
    ax.set_color_cycle(['black', 'red', 'blue'])
    ax.plot(dist, trained, 'o')
    iso = IsotonicRegression()
    iso.fit(dist, trained)
    scores = np.linspace(np.amin(dist), np.amax(dist), 10000)
    ax.plot(scores, iso.predict(scores))
    linear = LinearRegression()
    linear.fit(dist.reshape(-1, 1), trained.reshape(-1, 1))
    ax.plot(scores.reshape(-1, 1), linear.predict(scores.reshape(-1, 1)))
    return fig_score_differences


def plot_score_distributions(scores_pos, scores_neg, calibrator):
    fig_score_distributions = plt.figure('score_distributions')
    fig_score_distributions.clf()
    ax = plt.subplot(111)
    ax.set_color_cycle(['blue', 'red'])
    x, bins, p = ax.hist(scores_pos, color='blue', alpha=0.3, label=r'hist$^+$',
                         range=[0, 1], normed=True)
    for item in p:
        item.set_height(item.get_height() / np.amax(x))
    x, bins, p = ax.hist(scores_neg, color='red', alpha=0.3, label=r'hist$^-$',
                         range=[0, 1], normed=True)
    for item in p:
        item.set_height(item.get_height() / np.amax(x))
    al_pos, bt_pos = fit_beta_moments(scores_pos)
    al_pos = np.clip(al_pos, 1e-16, np.inf)
    bt_pos = np.clip(bt_pos, 1e-16, np.inf)
    al_neg, bt_neg = fit_beta_moments(scores_neg)
    al_neg = np.clip(al_neg, 1e-16, np.inf)
    bt_neg = np.clip(bt_neg, 1e-16, np.inf)
    scores = np.linspace(0.0, 1.0, 1000)
    pdf_pos = beta.pdf(scores, al_pos, bt_pos)
    p_pos = pdf_pos[pdf_pos < np.inf]
    if len(p_pos) == 0:
        p_pos = pdf_pos
    ax.plot(scores, pdf_pos/np.amax(p_pos), linestyle=":",
            label='p(x|+)')
    pdf_neg = beta.pdf(scores, al_neg, bt_neg)
    p_neg = pdf_neg[pdf_neg < np.inf]
    if len(p_neg) == 0:
        p_neg = pdf_neg
    ax.plot(scores, pdf_neg/np.amax(p_neg), linestyle=":",
            label='p(x|-)')

    prior_pos = len(scores_pos) / (len(scores_pos) + len(scores_neg))
    prior_neg = len(scores_neg) / (len(scores_pos) + len(scores_neg))
    denominator = pdf_pos * prior_pos + pdf_neg * prior_neg
    prob_pos = (pdf_pos * prior_pos) / denominator
    prob_neg = (pdf_neg * prior_neg) / denominator
    ax.plot(scores, prob_pos, linestyle="--", label=r'p(+|x)$_{separate}$')
    ax.plot(scores, prob_neg, linestyle="--", label=r'p(-|x)$_{separate}$')
    pr_pos = calibrator.predict(scores)
    pr_neg = 1.0 - pr_pos
    ax.plot(scores, pr_pos, label='p(+|x)$_{betacal}$')
    ax.plot(scores, pr_neg, label='p(-|x)$_{betacal}$')
    ax.set_xlim([-0.001, 1.001])
    ax.set_ylim([0, 2.1])
    ax.legend(loc='upper right')
    return fig_score_distributions


# def sigmoid(x):
#     return np.exp(x) / (1 + np.exp(x))
#
#
# if __name__ == '__main__':
#     from sklearn.linear_model import LogisticRegression
#     np.random.seed(42)
#     # Random scores
#     n = np.random.normal(loc=-4, scale=2, size=100)
#     p = np.random.normal(loc=4, scale=2, size=100)
#     s = np.append(n, p)
#     plt.hist(s)
#     plt.show()
#     s.sort()
#     s1 = s.reshape(-1, 1)
#
#     # Obtaining probabilities from the scores
#     s1 = sigmoid(s1)
#     # Obtaining the two features for beta-calibration with 3 parameters
#     s1 = np.log(np.hstack((s1, 1.0 - s1)))
#     # s1[:, 1] *= -1
#
#     # Generating random labels
#     y = np.append(np.random.binomial(1, 0.1, 40), np.random.binomial(1, 0.3,
#                                                                      40))
#     y = np.append(y, np.random.binomial(1, 0.4, 40))
#     y = np.append(y, np.random.binomial(1, 0.4, 40))
#     y = np.append(y, np.ones(40))
#
#     # Fitting Logistic Regression without regularization
#     lr = LogisticRegression(C=99999999999)
#     lr.fit(s1, y)
#
#     linspace = np.linspace(-10, 10, 100)
#     l = sigmoid(linspace).reshape(-1, 1)
#     l1 = np.log(np.hstack((l, 1.0 - l)))
#     # l1[:, 1] *= -1
#
#     probas = lr.predict_proba(l1)[:, 1]
#     s_exp = sigmoid(s)
#     fig_map = plot_niculescu_mizil_map([probas], [s_exp, y, l],
#                                        ['beta'], alpha=1)
#
#     plt.show()

