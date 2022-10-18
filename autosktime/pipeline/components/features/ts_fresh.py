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

    def __init__(self, config_dict: Dict[str, bool], config_id: ConfigId = None):
        super().__init__()
        self.config_dict = config_dict
        self.config_id = config_id

    def transform(self, X: np.ndarray) -> np.ndarray:
        config_context: ConfigContext = ConfigContext.instance()
        y = pd.DataFrame(config_context.get_config(self.config_id, key='y'))

        features = []
        fnames = []

        for fname, value in self.config_dict.items():
            if fname in TSFreshFeatureGenerator.features and value:
                fnames.append(fname)
                features.append(TSFreshFeatureGenerator.features[fname](X))
        Xt = pd.DataFrame(np.concatenate(features, axis=1))
        Xt = tsfresh.feature_selection.select_features(Xt, y[0])

        # Construct feature names
        columns = np.repeat(fnames, X.shape[2])
        for feat in range(X.shape[2]):
            columns[feat::X.shape[2]] = np.char.add(columns[feat::X.shape[2]], f'_{feat}')
        Xt.columns = columns[Xt.columns]

        return Xt.values

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        hps = [CategoricalHyperparameter(fname, [True, False]) for fname in TSFreshFeatureGenerator.features.keys()]

        cs = ConfigurationSpace()
        cs.add_hyperparameters(hps)

        return cs
