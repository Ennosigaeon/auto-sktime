from typing import Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sktime.datatypes import convert_to
from sktime.utils.plotting import plot_series as plot_series_

from autosktime.constants import SUPPORTED_Y_TYPES


def plot_series(
        y_train: Optional[SUPPORTED_Y_TYPES],
        y_test: Optional[SUPPORTED_Y_TYPES],
        y_pred: Optional[SUPPORTED_Y_TYPES],
        ax=None
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    if isinstance(y_pred.index, pd.MultiIndex):
        series = []
        labels = []
        for idx in y_pred.index.levels[0]:
            for y in (y_train, y_test, y_pred):
                if y is not None:
                    series.append(y.loc[idx])
            if y_train is not None:
                labels.append(f'y_train_{idx}')
            if y_test is not None:
                labels.append(f'y_test_{idx}')
            if y_pred is not None:
                labels.append(f'y_pred_{idx}')
    else:
        series = [y_train, y_test, y_pred]
        labels = ['y_train', 'y_test', 'y_pred']
    return plot_series_(*series, markers=[''] * len(series), labels=labels, ax=ax)


def plot_grouped_series(
        y_train: Optional[SUPPORTED_Y_TYPES],
        y_test: Optional[SUPPORTED_Y_TYPES],
        y_pred: Optional[SUPPORTED_Y_TYPES],
        ax=None
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    if isinstance(y_pred.index, pd.MultiIndex):
        series = []
        labels = []
        for idx in y_pred.index.levels[0]:
            t = []
            for y in (y_train, y_test, y_pred):
                if y is not None:
                    t.append(convert_to(y.loc[idx], "pd.Series", "Series"))
            series.append(t)

            count = ['_nolegend_' for x in (y_train, y_test, y_pred) if x is not None]
            count[0] = f'y_{idx}'
            labels.append(count)
    else:
        series = [[y_train, y_test, y_pred]]
        labels = [['y', None, None]]

    if ax is None:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25), dpi=300)

    colors = sns.color_palette("colorblind", n_colors=len(series))

    for y, color, label in zip(series, colors, labels):
        for i in range(len(y)):
            sns.lineplot(x=np.arange(y[i].shape[0]), y=y[i], ax=ax, label=label[i], color=color)

    return ax
