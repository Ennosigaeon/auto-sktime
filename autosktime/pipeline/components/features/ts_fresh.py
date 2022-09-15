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

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = [
            latest(X),
            mean(X),
            variance(X),
            standard_deviation(X),
            length(X),
            skewness(X),
            median(X),
            kurtosis(X),
            crest(X),
            minimum(X),
            maximum(X),
            last_location_of_maximum(X),
            mean_second_derivative_central(X),
            abs_energy(X),
            first_location_of_minimum(X),
            last_location_of_minimum(X),
            first_location_of_maximum(X),
            last_location_of_maximum(X),
            mean_abs_change(X),
            mean_change(X),
            sum_values(X),
            absolute_sum_of_changes(X),
            variance_larger_than_standard_deviation(X),
            percentage_of_reoccurring_values_to_all_values(X),
            has_duplicate(X),
            count_above_mean(X),
            count_below_mean(X),
            ratio_beyond_r_sigma(X, 1),
            large_standard_deviation(X, 1),
            quantile(X, 0.5),
            number_crossing_m(X, 5),
            count_below(X, 5),
            count_above(X, 5),
            cid_ce(X, False),
            range_count(X, 0, 10),
            c3(X, 2),
            symmetry_looking(X, 2),
            time_reversal_asymmetry_statistic(X, 2),
            fft_coefficient(X, 2, 'real'),
            autocorrelation(X, 2),
            number_peaks(X, 2),
            index_mass_quantile(X, 0.5),
        ]
        Xt = np.concatenate(features, axis=1)
        return Xt
