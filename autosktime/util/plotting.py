from typing import Tuple, Union, Optional

import matplotlib.pyplot as plt
import pandas as pd
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
