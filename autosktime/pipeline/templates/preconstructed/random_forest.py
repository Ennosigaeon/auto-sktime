import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UnParametrizedHyperparameter
from autosktime.data import DatasetProperties
from autosktime.data.benchmark import PHME20Benchmark
from autosktime.pipeline.components.base import AutoSktimeComponent, UpdatablePipeline, \
    SwappedInput, AutoSktimePreprocessingAlgorithm, COMPONENT_PROPERTIES
from autosktime.pipeline.components.features.flatten import FlatteningFeatureGenerator
from autosktime.pipeline.components.reduction.panel import RecursivePanelReducer
from autosktime.pipeline.components.regression.random_forest import RandomForestComponent
from autosktime.pipeline.templates import PanelRegressionPipeline
from autosktime.pipeline.templates.base import get_pipeline_search_space
from autosktime.pipeline.templates.preconstructed import find_benchmark_settings


class DataScaler(AutoSktimePreprocessingAlgorithm):

    def __init__(self):
        super().__init__()
        self.profiles = []
        self.scalers = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        settings = find_benchmark_settings(X)

        self.scalers = {}
        self.profiles = X['Kmeans_Profile'].unique()
        # Full dataset fit
        for profile in self.profiles:
            sensors_readings = X[(X['Kmeans_Profile'] == profile)].filter(
                settings.sensor_names + settings.artificial_names)  # Get sensor readings
            state_scaler = StandardScaler().fit(sensors_readings)  # Fit scaler
            self.scalers[profile] = state_scaler  # Add to sclaer_list for further reference

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        settings = find_benchmark_settings(X)

        # Full dataset transform
        for profile in self.profiles:
            sensors_readings = X[(X['Kmeans_Profile'] == profile)].filter(
                settings.sensor_names + settings.artificial_names)  # Get sensor readings
            if sensors_readings.shape[0] == 0:
                continue  # No matching profiles found in X_df
            cols = sensors_readings.columns
            normalized_sensor_readings = self.scalers[profile].transform(sensors_readings)  # transform sensor readings
            X.loc[(X['Kmeans_Profile'] == profile), cols] = normalized_sensor_readings  # record transformed values

        return X

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        pass


class FeatureGeneration(AutoSktimePreprocessingAlgorithm):

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        settings = find_benchmark_settings(X)
        if settings.name == PHME20Benchmark.name():
            X = X.copy()
            X['Pressure_Drop'] = X['Upstream_Pressure(psi)'] - X['Downstream_Pressure(psi)']
            # Reorder columns
            X = X[['Flow_Rate(ml/m)', 'Upstream_Pressure(psi)', 'Downstream_Pressure(psi)', 'Pressure_Drop',
                   'Particle Size (micron)', 'Solid Ratio(%)', 'Kmeans_Profile']]

        return X

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        pass


class OperationCondition(AutoSktimePreprocessingAlgorithm):

    def fit(self, X: pd.DataFrame, y: pd.Series):
        settings = find_benchmark_settings(X)

        sample_df = X.sample(frac=0.05)
        X_train_for_kmeans = sample_df[settings.setting_names].values
        estimator = KMeans(n_clusters=settings.n_profiles, max_iter=10, random_state=self.random_state)
        estimator.fit_predict(X_train_for_kmeans)
        self.estimator = estimator

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        settings = find_benchmark_settings(X)
        X = X.copy()
        # noinspection PyUnresolvedReferences
        X['Kmeans_Profile'] = self.estimator.predict(X[settings.setting_names].values)
        return X

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        pass


class FixedRandomForest(RandomForestComponent):

    def get_max_iter(self):
        return 100

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        max_features = UnParametrizedHyperparameter('max_features', 1.0)
        max_depth = UnParametrizedHyperparameter('max_depth', 25)
        min_samples_split = UnParametrizedHyperparameter('min_samples_split', 2)
        min_samples_leaf = UnParametrizedHyperparameter('min_samples_leaf', 2)
        bootstrap = UnParametrizedHyperparameter('bootstrap', 'True')
        n_jobs = UnParametrizedHyperparameter('n_jobs', 4)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, n_jobs])
        return cs


class FixedRecursivePanelReducer(RecursivePanelReducer):

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        window_length = CategoricalHyperparameter('window_length', [15])
        step_size = UnParametrizedHyperparameter('step_size', 0.001)

        estimator = get_pipeline_search_space(self.estimator.steps, dataset_properties=dataset_properties)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([window_length, step_size])
        cs.add_configuration_space('estimator', estimator)

        if self.transformers is not None:
            transformers = ConfigurationSpace()
            for name, t in self.transformers:
                transformers.add_configuration_space(name, t.get_hyperparameter_search_space(dataset_properties))
            cs.add_configuration_space('transformers', transformers)

        return cs


class RandomForestPipeline(PanelRegressionPipeline):
    """
    Implementation of "Remaining Useful Life Prediction for Experimental Filtration System: A Data Challenge"
    Source code adapted from https://github.com/zakkum42/phme20-public
    Disable pynisher (use_pynisher) and multi-fidelity approximations (use_multi_fidelity) when selecting this template.
    Furthermore, use a short timeout as the config space contains only a single configuration (see https://github.com/automl/SMAC3/issues/21)
    """

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        pipeline = UpdatablePipeline(steps=[
            ('flatten', FlatteningFeatureGenerator(random_state=self.random_state)),
            ('regression', FixedRandomForest(random_state=self.random_state))
        ])

        steps = [
            ('reduction',
             FixedRecursivePanelReducer(
                 transformers=[
                     ('operation_condition', SwappedInput(OperationCondition(random_state=self.random_state))),
                     ('feature_selection', SwappedInput(FeatureGeneration())),
                     ('scaling', SwappedInput(DataScaler())),
                 ],
                 estimator=pipeline,
                 random_state=self.random_state,
                 dataset_properties=self.dataset_properties)
             ),
        ]
        return steps

    def _get_hyperparameter_search_space(self) -> ConfigurationSpace:
        return super()._get_hyperparameter_search_space()
