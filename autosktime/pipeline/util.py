from typing import Any, Dict, Tuple, Optional, List

import pandas as pd
from sktime.datatypes import check_is_scitype


class NotVectorizedMixin:

    def _check_X_y(self, X=None, y=None, return_metadata: bool = False):
        metadata = dict()

        if X is None and y is None:
            if return_metadata:
                return None, None, metadata
            else:
                return None, None

        ALLOWED_SCITYPES = ["Series", "Panel", "Hierarchical"]

        # checking X
        if X is not None:
            X_valid, _, X_metadata = check_is_scitype(
                X, scitype=ALLOWED_SCITYPES, return_metadata=True, var_name="X"
            )

            metadata["_X_mtype_last_seen"] = X_metadata['mtype']
            metadata["_X_input_scitype"] = X_metadata['scitype']
            metadata["_convert_case"] = "case 1: scitype supported"

        # checking y
        if y is not None:
            y_valid, _, y_metadata = check_is_scitype(
                y, scitype=ALLOWED_SCITYPES, return_metadata=True, var_name="y"
            )
            msg = (
                "y must be in an sktime compatible format, "
                "of scitype Series, Panel or Hierarchical, "
                "for instance a pandas.DataFrame with sktime compatible time indices, "
                "or with MultiIndex and lowest level a sktime compatible time index. "
                "See the forecasting tutorial examples/01_forecasting.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb"
            )
            if not y_valid:
                raise TypeError(msg)

            self._y_mtype_last_seen = y_metadata["mtype"]

        if return_metadata:
            return X, y, metadata
        else:
            return X, y


def sub_configuration(params: Dict[str, Any], init_params: Dict[str, Any]) -> Tuple[str, Dict[str, any]]:
    new_params = {}
    choice = params['__choice__']

    for param, value in params.items():
        if param == '__choice__':
            continue

        param = param.replace(f'{choice}:', '', 1)
        new_params[param] = value

    if init_params is not None:
        for param, value in init_params.items():
            param = param.replace(f'{choice}:', '', 1)
            new_params[param] = value

    return choice, new_params


def frequency_to_sp(freq: Optional[pd.offsets.BaseOffset]) -> List[int]:
    if freq is None or freq.name is None:
        return [1, 4, 5, 7, 12, 24, 30, 96, 365]

    if freq.name in ('D', 'C'):
        return [7, 1, 30, 96, 365]
    elif freq.name in ('B',):
        return [5, 1, 20]
    elif freq.name in ('BMS', 'BM', 'CBMS', 'CBM', 'MS', 'M', 'LWOM-MON'):
        return [12, 1, 4]
    elif freq.name in ('SM-15', 'SMS-15'):
        return [6, 1, 2]
    elif freq.name in ('BH', 'CBH', 'H'):
        return [24, 1, 4, 6, 12]
    elif freq.name in ('AS-JAN', 'BAS-JAN', 'A-DEC', 'BA-DEC', 'RE-N-JAN-MON', 'W'):
        return [52, 1]
    elif freq in ('QS-MAR', 'BQS-MAR', 'Q-MAR', 'BQ-MAR', 'REQ-N-JAN-MON-1', 'WOM-1MON'):
        return [4, 1]
    elif freq.name in ('T',):
        return [60, 1]
    elif freq.name in ('S',):
        return [60, 1, 3600]
    elif freq.name in ('L', 'U', 'N'):
        return [1000, 1]
    else:
        return [1, 4, 5, 7, 12, 24, 365]


# noinspection PyUnresolvedReferences
Int64Index = pd.core.indexes.numeric.Int64Index
