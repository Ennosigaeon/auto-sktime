import numpy as np
import pandas as pd
import tsfresh.feature_selection
from inspect import getmembers, isfunction
from typing import Dict

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.features import BaseFeatureGenerator, _features
from autosktime.util.backend import ConfigId, ConfigContext

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

    def __init__(self, config_dict: Dict[str, bool]):
        super().__init__()
        self.config_dict = config_dict

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = []
        for fname, value in self.config_dict.items():
            if fname not in TSFreshFeatureGenerator.features or not value:
                continue
            features.append(TSFreshFeatureGenerator.features[fname](X))
        Xt = np.concatenate(features, axis=1)
        return Xt

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        hps = [CategoricalHyperparameter(fname, [True, False]) for fname in TSFreshFeatureGenerator.features.keys()]

        cs = ConfigurationSpace()
        cs.add_hyperparameters(hps)

        return cs
