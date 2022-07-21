from typing import Tuple, List

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES, UpdatablePipeline, \
    SwappedInput
from autosktime.pipeline.components.data_preprocessing import DataPreprocessingPipeline
from autosktime.pipeline.components.downsampling.elimination import EliminationDownSampler
from autosktime.pipeline.components.index import AddIndexComponent
from autosktime.pipeline.components.preprocessing.impute import ImputerComponent
from autosktime.pipeline.components.reduction.panel import RecursivePanelReducer
from autosktime.pipeline.components.regression import RegressorChoice
from autosktime.pipeline.templates.base import ConfigurableTransformedTargetForecaster
from autosktime.pipeline.util import NotVectorizedMixin, Int64Index


class PanelRegressionPipeline(NotVectorizedMixin, ConfigurableTransformedTargetForecaster):
    _tags = {
        'scitype:transform-input': 'Panel',
        'scitype:transform-output': 'Panel',
        "scitype:y": "both",
        "y_inner_mtype": 'pd.DataFrame',
        "X_inner_mtype": 'pd.DataFrame',
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
        'X-y-must-have-same-index': True,
        "vectorize_panel_data": True,
        'fit_is_empty': False
    }

    def fit(self, y, X=None, fh=None):
        # mean = y.groupby(y.index.get_level_values(1)).mean()
        res = super().fit(y, X, fh)

        panel_size = y.groupby(level=0).size()
        self._identical_fh = np.all(panel_size == panel_size.iloc[0])

        return res

    def _predict(self, fh: ForecastingHorizon = None, X: pd.DataFrame = None):
        if X is None:
            raise ValueError('Provide a timeseries that should be forecasted as exogenous data')

        if isinstance(X.index, pd.MultiIndex):
            y_pred_complete = []

            index = X.index.remove_unused_levels()
            for key in index.levels[0]:
                if fh is not None and not self._identical_fh:
                    if fh.is_relative:
                        last_observation = X.xs(key, level=0).index[-1]
                        fh_ = ForecastingHorizon(fh.to_pandas() + last_observation, is_relative=False)
                    else:
                        fh_ = ForecastingHorizon(X.xs(key, level=0).index, is_relative=False)
                else:
                    fh_ = fh

                X_ = X.loc[[key]]

                y_pred = super()._predict(fh_, X=X_)
                y_pred_complete.append(y_pred)

            y_pred = pd.concat(y_pred_complete)
        else:
            y_pred = super()._predict(fh, X=X)

        return y_pred
        # return super()._predict(fh, X)

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        pipeline = UpdatablePipeline(steps=[
            ('preprocessing', DataPreprocessingPipeline(random_state=self.random_state)),
            ('regression', RegressorChoice(random_state=self.random_state))
        ])

        steps = [
            # Detrending by collapsing panel data to univariate timeseries by averaging
            ('imputation', ImputerComponent(random_state=self.random_state)),
            ('reduction',
             RecursivePanelReducer(
                 transformers=[
                     ('add_index', SwappedInput(AddIndexComponent())),
                     ('downsampling', EliminationDownSampler(random_state=self.random_state))
                 ],
                 estimator=pipeline,
                 random_state=self.random_state,
                 step_size=1,
                 concat_multiindex=True,
                 dataset_properties=self.dataset_properties)
             ),
        ]
        return steps

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: False,
            HANDLES_MULTIVARIATE: False,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
