from typing import Hashable

import pandas as pd
from sktime.forecasting.model_selection._split import ACCEPTED_Y_TYPES


def get_name(y: ACCEPTED_Y_TYPES) -> Hashable:
    if isinstance(y, pd.Series):
        return y.name
    elif isinstance(y, pd.DataFrame):
        if len(y.columns) != 1:
            raise ValueError('Expected y to have exactly 1 column. Multivariate timeseries currently not supported')
        return y.columns[0]
    else:
        # TODO
        raise ValueError('')


def resolve_index(y_index: pd.Index) -> pd.Index:
    if isinstance(y_index, pd.MultiIndex):
        # TODO we assume that all panel data have the same fh
        y_index = y_index.remove_unused_levels().levels[-1]
    return y_index
