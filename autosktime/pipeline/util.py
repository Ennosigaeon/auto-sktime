from typing import Any, Dict, Tuple

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


# noinspection PyUnresolvedReferences
Int64Index = pd.core.indexes.numeric.Int64Index
