import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.utils.fixes import parse_version
import matplotlib



def dispaly_cluster_dendrogram(df):
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform


    cat_mask = df.dtypes.isin(['object', 'category',
                               'bool', 'datetime'])

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    corr = df.loc[:, ~cat_mask].corr().values
    corr = (corr + corr.T) / 2

    dist_mat = 1 - np.abs(corr)
    dist_link = np.abs(hierarchy.linkage(squareform(dist_mat), method='ward'))
    dendro = hierarchy.dendrogram(dist_link,
                                  labels=df.loc[:, ~cat_mask].columns.to_list(),
                                  ax=ax, leaf_rotation=90)
    return fig, dendro


def plot_permutation_importance(clf, X, y, ax):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    # `labels` argument in boxplot is deprecated in matplotlib 3.9 and has been
    # renamed to `tick_labels`. The following code handles this, but as a
    # scikit-learn user you probably can write simpler code by using `labels=...`
    # (matplotlib < 3.9) or `tick_labels=...` (matplotlib >= 3.9).
    tick_labels_parameter_name = (
        "tick_labels"
        if parse_version(matplotlib.__version__) >= parse_version("3.9")
        else "labels"
    )
    tick_labels_dict = {tick_labels_parameter_name: X.columns[perm_sorted_idx]}
    ax.boxplot(result.importances[perm_sorted_idx].T, vert=False,
               **tick_labels_dict)
    ax.axvline(x=0, color="k", linestyle="--")
    return ax


def plot_gini_permutation_importance(model, X, y):
    try:
        mdi_importances = pd.Series(model.feature_importances_,
                                    index=X.columns)
        tree_importance_sorted_idx = np.argsort(model.feature_importances_)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
        mdi_importances.sort_values().plot.barh(ax=ax1, rot=45)
        ax1.set_xlabel("Gini importance")
        plot_permutation_importance(model, X, y, ax2)
        ax2.set_xlabel("Decrease in accuracy score")
        fig.suptitle(
            "Impurity-based vs. permutation importances on multicollinear\
features (train set)"
        )
        _ = fig.tight_layout()
    except Exception as e:
        print(e)
