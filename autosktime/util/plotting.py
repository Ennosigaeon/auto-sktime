from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from sktime.utils.plotting import plot_series as plot_series_

from autosktime.constants import SUPPORTED_Y_TYPES


def plot_series(
        y_train: SUPPORTED_Y_TYPES,
        y_test: SUPPORTED_Y_TYPES,
        y_pred: SUPPORTED_Y_TYPES,
        ax=None
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    if isinstance(y_pred.index, pd.MultiIndex):
        series = []
        labels = []
        for idx in y_pred.index.levels[0]:
            for y in (y_train, y_test, y_pred):
                series.append(y.loc[idx])
            labels += [f'y_train_{idx}', f'y_test_{idx}', f'y_pred_{idx}']
    else:
        series = [y_train, y_test, y_pred]
        labels = ['y_train', 'y_test', 'y_pred']
    return plot_series_(*series, labels=labels, ax=ax)
