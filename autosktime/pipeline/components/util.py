from sktime.datatypes import check_is_scitype


class NotVectorizedMixin:

    def _check_X_y(self, X=None, y=None):
        if X is None and y is None:
            return None, None

        ALLOWED_SCITYPES = ["Series", "Panel", "Hierarchical"]

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

        return X, y
