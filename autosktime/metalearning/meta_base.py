import logging
import os.path
import pathlib
import pickle
import typing
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration

from autosktime.constants import TASK_TYPES_TO_STRING
from autosktime.metalearning.kND import KNearestDataSets
from autosktime.smac.prior import Prior, KdePrior, UniformPrior


class MetaBase:
    def __init__(
            self,
            configuration_space: ConfigurationSpace,
            task: int,
            metric: str,
            distance_measure: str = 'ddtw',
            base_dir: str = None,
            logger: logging.Logger = None
    ):
        self.configuration_space = configuration_space
        self.task = task
        self.metric = metric
        self.distance_measure = distance_measure

        if base_dir is None:
            base_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), 'files')
        self.base_dir = base_dir

        if logger is None:
            logger = logging.Logger('meta-base')
        self.logger = logger

        self._kND: Optional[KNearestDataSets] = None
        self._timeseries, self._configs = self._load_instances()

    def _load_instances(self) -> Tuple[typing.Dict[str, pd.Series], pd.DataFrame]:
        folder = os.path.join(self.base_dir, f'{TASK_TYPES_TO_STRING[self.task]}-{self.metric}')
        with open(os.path.join(folder, 'timeseries.npy.gz'), 'rb') as f:
            timeseries = pickle.load(f)
        configs = pd.read_csv(os.path.join(folder, 'configurations.csv')).convert_dtypes()
        return timeseries, configs

    def suggest_configs(
            self,
            y: pd.Series,
            num_initial_configurations: int,
            exclude_double_configurations: bool = True
    ) -> List[Configuration]:
        neighbors, _ = self._get_neighbors(y)

        added_configurations = set()
        configurations = []
        for neighbor in neighbors:
            try:
                config = self._get_configuration(neighbor, index=0)
                if not exclude_double_configurations or config not in added_configurations:
                    added_configurations.add(config)
                    configurations.append(config)
            except KeyError:
                self.logger.warning(f'Best configuration for {neighbor} not found')
                continue

        return configurations[:num_initial_configurations]

    def _get_neighbors(self, y: pd.Series, k: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        if self._kND is None:
            self._kND = KNearestDataSets(metric=self.distance_measure, logger=self.logger)
            self._kND.fit(self._timeseries)

        names, distances = self._kND.kneighbors(y, k=k)
        mask = ~np.isinf(distances) & (distances < 10000)
        return names[mask], distances[mask]

    def _get_configuration(self, dataset: str, index: int) -> Configuration:
        dataset_configs = self._configs.loc[
            self._configs['__dataset__'] == dataset,
            self._configs.columns.difference(('__dataset__', '__loss__'))
        ].sort_values(by='__loss__')

        return Configuration(self.configuration_space, vector=dataset_configs.iloc[index].values)

    def suggest_univariate_prior(self, y: pd.Series, num_datasets: int, cutoff: float = 0.2) -> Dict[str, Prior]:
        neighbors, distances = self._get_neighbors(y, num_datasets)

        if len(neighbors) == 0:
            return {}

        configs = []
        for neighbor, distance in zip(neighbors, distances):
            df = self._get_configuration_array(neighbor, cutoff)
            df['weights'] = distance
            configs.append(df)
        df = pd.concat(configs)

        # Normalize weights
        df['weights'] = 1 - df['weights'] / np.linalg.norm(df['weights'])
        df['weights'] /= df['weights'].sum()

        priors = {}
        for hp_name in df:
            if hp_name == 'weights':
                continue
            try:
                hp = self.configuration_space.get_hyperparameter(hp_name)

                observations = df[hp_name]
                filled_values = (~pd.isna(observations)).sum()
                if filled_values < 2:
                    prior = UniformPrior(hp)
                else:
                    try:
                        prior = KdePrior(hp, observations, weights=df['weights'])
                    except RuntimeError as ex:
                        self.logger.warning(f'Failed to fit KdePrior: `{ex}`. Using UniformPrior as fallback')
                        prior = UniformPrior(hp)

                priors[hp_name] = prior
            except KeyError:
                continue
        return priors

    def _get_configuration_array(self, dataset: str, cutoff: float) -> pd.DataFrame:
        hp_names = self._configs.columns.difference(['__dataset__', '__loss__'])
        dataset_configs = self._configs.loc[
            self._configs['__dataset__'] == dataset,
            hp_names
        ]
        if dataset_configs.shape[0] == 0:
            return dataset_configs

        max_index = max(int(cutoff * dataset_configs.shape[0]), 1)
        return dataset_configs.iloc[:max_index, :]
