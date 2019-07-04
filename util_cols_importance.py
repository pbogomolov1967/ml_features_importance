# helpers to calc features / cols importance

import matplotlib.pyplot as plt
import numpy as np


def collapse_values(importance, features):
    """collapse cols w/ values (A_A, A_B, A_C for example into just A w/ sum(A_weights))"""
    assert len(importance) == len(features)
    assert abs(sum(importance) - 1) < 1e-10
    cols = set([f.split('_')[0] for f in features])
    importance2, features2 = [], []
    for c in cols:
        pairs = zip(features, importance)
        w = sum([w for (f, w) in pairs if f.split('_')[0] == c])
        features2.append(c)
        importance2.append(w)

    indices = np.argsort(importance2)[::-1]  # sort by descending order
    importance2 = [importance2[ix] for ix in indices]  # should be in descending order
    features2 = [features2[ix] for ix in indices]
    assert abs(sum(importance2) - 1) < 1e-10

    return importance2, features2


def calc_importance(features, feature_importance_weights, show_prc, collapse_vals=True):
    """show only top part of most important features"""
    importance = feature_importance_weights
    indices = np.argsort(importance)[::-1]  # sort by descending order
    importance_sorted = importance[indices]  # should be in descending order
    features2 = [features[ix] for ix in indices]

    if collapse_vals:
        importance_sorted, features2 = collapse_values(importance_sorted, features2)

    def map_feature(col):
        prefix = col
        suffix = ''

        max_len = 40
        if len(prefix) + len(suffix) > max_len:
            max_len -= len(suffix) - 3
            prefix = prefix[:max_len] + '..'

        return ''.join([prefix, suffix])

    features2 = list(map(map_feature, features2))

    if show_prc > 0:
        part = round(len(feature_importance_weights) * show_prc / 100)
        importance_part = importance_sorted[:part]
        features_part = features2[:part]

        plt.figure(1).canvas.set_window_title('Feature selection')
        plt.subplots_adjust(left=0.5)
        plt.title(f'Feature Importance (top {show_prc}%)' if show_prc < 100 else 'Feature Importance')
        plt.barh(range(len(importance_part)), importance_part[::-1], color='b', align='center')
        plt.yticks(range(len(importance_part)), features_part[::-1])
        plt.xlabel('Relative Importance')
        plt.show()

    return features2, importance_sorted


###