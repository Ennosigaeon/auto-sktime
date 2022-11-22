import warnings

import numpy as np
import pandas as pd
import tsfresh.feature_selection
from inspect import getmembers, isfunction
from typing import Dict

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.features import BaseFeatureGenerator, _features
from autosktime.util.backend import ConfigId

"binned_entropy"
"energy_ratio_by_chunks"
"linear_trend"
"spkt_welch_density"
"agg_autocorrelation"
"fft_aggregated"
"partial_autocorrelation"
"approximate_entropy"
"number_cwt_peaks"
"augmented_dickey_fuller"
"agg_linear_trend"
"max_langevin_fixed_point"
"friedrich_coefficients"
"cwt_coefficients"
"ar_coefficient"
"change_quantiles"
"longest_strike_below_mean"
"longest_strike_above_mean"
"sum_of_reoccurring_data_points"
"sum_of_reoccurring_values"
"sample_entropy"


class TSFreshFeatureGenerator(BaseFeatureGenerator):
    features = dict(getmembers(_features, isfunction))

    def __init__(self, config_dict: Dict[str, bool], config_id: ConfigId = None):
        super().__init__()
        self.config_dict = config_dict
        self.config_id = config_id

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xt = self._calc_features(X)
        Xt = Xt.dropna(axis=1)

        column_names = Xt.columns
        Xt.columns = np.arange(len(column_names))
        Xt = tsfresh.feature_selection.select_features(Xt, pd.Series(y))
        Xt.columns = column_names[Xt.columns]
        self._selected_columns = np.copy(Xt.columns.values)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = self._calc_features(X)
        Xt = Xt[self._selected_columns]
        return Xt.values

    def _calc_features(self, X: np.ndarray) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            features = []
            fnames = []

            for fname, value in self.config_dict.items():
                if fname in TSFreshFeatureGenerator.features and value:
                    fnames.append(fname)
                    features.append(TSFreshFeatureGenerator.features[fname](X))
            Xt = pd.DataFrame(np.concatenate(features, axis=1))

            # Construct feature names
            columns = np.repeat(fnames, X.shape[2])
            for feat in range(X.shape[2]):
                columns[feat::X.shape[2]] = np.char.add(columns[feat::X.shape[2]], f'_{feat}')
            Xt.columns = columns

            return Xt

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        hps = [CategoricalHyperparameter(fname, [True, False]) for fname in TSFreshFeatureGenerator.features.keys()]

        cs = ConfigurationSpace()
        cs.add_hyperparameters(hps)

        return cs
