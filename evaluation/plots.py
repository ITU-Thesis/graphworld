import matplotlib.pyplot as plt
import seaborn as sns
from constants import TRAINING_SCHEMES, TEST_METRIC
import pandas as pd
import numpy as np

def plot_training_scheme_histograms(df):
    sns.set()
    N_bins = 20
    results = []
    for key in TRAINING_SCHEMES:
        v = df[df.Training_scheme == key]
        bins = pd.cut(v[TEST_METRIC], bins=np.linspace(1/2, 1, N_bins)).value_counts().reset_index().rename(columns={ 'index': 'ROC-AUC', 'test_rocauc_ovr': 'Fraction' })
        bins['Fraction'] = bins['Fraction'] / bins['Fraction'].sum()
        bins['Training_scheme'] = key
        bins['ROC-AUC'] = bins['ROC-AUC'].apply(lambda x: f'({x.left:.2f}, {x.right:.2f}]')
        results += [bins]
    bin_df = pd.concat(results)
    bin_df.head()
    g = sns.FacetGrid(data=bin_df, col='Training_scheme', height=8, aspect=1)
    g.map(sns.barplot, 'ROC-AUC', 'Fraction')

    plt.tight_layout()
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax.set_title(ax.get_title().replace('Training_scheme=', ''), fontsize=20)
    plt.tight_layout()


