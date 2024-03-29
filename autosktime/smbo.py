import copy
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Type, Any

import dask.distributed
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric
from smac import Callback, AlgorithmConfigurationFacade, MultiFidelityFacade
from smac.acquisition.function import AbstractAcquisitionFunction
from smac.facade import AbstractFacade
from smac.initial_design import DefaultInitialDesign, RandomInitialDesign, AbstractInitialDesign
from smac.intensifier import SuccessiveHalving
from smac.intensifier.intensifier import Intensifier
from smac.runhistory.dataclasses import TrajectoryItem
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario

from autosktime.automl_common.common.utils.backend import Backend
from autosktime.constants import Budget
from autosktime.data import DataManager
from autosktime.data.splitter import BaseSplitter
from autosktime.evaluation import ExecuteTaFunc
from autosktime.metalearning.meta_base import MetaBase
from autosktime.metrics import METRIC_TO_STRING, get_cost_of_crash
from autosktime.smac.acquisition import PriorAcquisitionFunction
from autosktime.smac.prior import Prior


class IntensifierGenerator:

    def __call__(self, *args, **kwargs) -> AbstractFacade:
        pass

    @staticmethod
    def _get_initial_design(scenario: Scenario, initial_configurations: List[Configuration]) -> AbstractInitialDesign:
        if initial_configurations is not None and len(initial_configurations) > 0:
            initial_design = DefaultInitialDesign(
                scenario=scenario,
                n_configs=0,
                additional_configs=initial_configurations,
            )
        else:
            initial_design = RandomInitialDesign(scenario=scenario, n_configs=10)
        return initial_design

    @staticmethod
    def _get_acq_function(scenario: Scenario, priors: Optional[Dict[str, Prior]]) -> AbstractAcquisitionFunction:
        acquisition_function = AlgorithmConfigurationFacade.get_acquisition_function(scenario)
        if priors is not None:
            acquisition_function = PriorAcquisitionFunction(acquisition_function, decay_beta=10, priors=priors)
        return acquisition_function


class SimpleIntensifierGenerator(IntensifierGenerator):

    def __call__(
            self,
            scenario: Scenario,
            seed: int,
            ta_kwargs: Dict,
            initial_configurations: List[Configuration],
            priors: Optional[Dict[str, Prior]],
            callbacks: List[Callback]
    ) -> AbstractFacade:
        ta_kwargs['scenario'] = scenario

        return AlgorithmConfigurationFacade(
            scenario=scenario,
            target_function=ExecuteTaFunc(**ta_kwargs),
            intensifier=Intensifier(scenario),
            initial_design=SimpleIntensifierGenerator._get_initial_design(scenario, initial_configurations),
            callbacks=callbacks,
            acquisition_function=SimpleIntensifierGenerator._get_acq_function(scenario, priors)
        )


class SHIntensifierGenerator(IntensifierGenerator):

    # Use eta and initial_budget as (2.0, 50.0) for 2 candidates per iteration with 50% elimination
    def __init__(self, budget_type: Budget, eta: int = 4.0):
        self.budget_type = budget_type
        self.eta = eta

    def __call__(
            self,
            scenario: Scenario,
            seed: int,
            ta_kwargs: Dict,
            initial_configurations: List[Configuration],
            priors: Optional[Dict[str, Prior]],
            callbacks: List[Callback],
    ) -> AbstractFacade:
        ta_kwargs['scenario'] = scenario
        ta_kwargs['budget_type'] = self.budget_type

        return MultiFidelityFacade(
            scenario=scenario,
            target_function=ExecuteTaFunc(**ta_kwargs),
            intensifier=SuccessiveHalving(scenario, eta=self.eta, seed=seed),
            initial_design=SHIntensifierGenerator._get_initial_design(scenario, initial_configurations),
            callbacks=callbacks,
            acquisition_function=SHIntensifierGenerator._get_acq_function(scenario, priors)
        )


class AutoMLSMBO:

    def __init__(
            self,
            config_space: ConfigurationSpace,
            datamanager: DataManager,
            backend: Backend,
            total_walltime_limit: float,
            func_eval_time_limit: float,
            runcount_limit: int,
            memory_limit: int,
            n_jobs: int,
            dask_client: dask.distributed.Client,
            metric: BaseForecastingErrorMetric,
            splitter: BaseSplitter,
            intensifier_generator: Type[IntensifierGenerator] = SimpleIntensifierGenerator,
            intensifier_generator_kwargs: Dict[str, Any] = None,
            use_pynisher: bool = True,
            refit: bool = False,
            initial_budget: float = 5.0,
            seed: int = 1,
            random_state: np.random.RandomState = None,
            metadata_directory: str = None,
            num_metalearning_configs: int = -1,
            hp_priors: bool = False,
            verbose: bool = False,
            ensemble_callback: Optional[Callback] = None,
            trials_callback: Optional[Callback] = None
    ):
        # data related
        self.datamanager = datamanager
        self.metric = metric
        self.backend = backend

        # the configuration space
        self.config_space = config_space

        intensifier_generator_kwargs = {} if intensifier_generator_kwargs is None else intensifier_generator_kwargs
        # noinspection PyArgumentList
        self.intensifier_generator = intensifier_generator(**intensifier_generator_kwargs)

        # Parallelization
        self.n_jobs = n_jobs
        self.dask_client = dask_client

        # Evaluation
        self.splitter = splitter
        self.use_pynisher = use_pynisher
        self.refit = refit
        self.initial_budget = initial_budget

        # and a bunch of useful limits
        self.worst_possible_result = get_cost_of_crash(self.metric)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.runcount_limit = runcount_limit
        self.memory_limit = memory_limit
        self.seed = seed
        self.random_state = random_state

        # metalearning
        self.metadata_directory = metadata_directory
        self.num_metalearning_configs = num_metalearning_configs
        self.hp_priors = hp_priors

        self.ensemble_callback = ensemble_callback
        self.trials_callback = trials_callback

        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        self.config_space.seed(self.seed)
        self.smac = self._get_smac()

    def optimize(self) -> Tuple[RunHistory, List[TrajectoryItem]]:
        # Main optimization loop
        self.smac.optimize()

        self.runhistory = self.smac.runhistory
        self.trajectory = self.smac.intensifier.trajectory

        return self.runhistory, self.trajectory

    def _get_smac(self, initial_budget: float = 5.0):
        scenario = Scenario(
            configspace=self.config_space,
            name="run",
            output_directory=Path(self.backend.get_smac_output_directory()),
            deterministic=True,
            crash_cost=self.worst_possible_result,
            termination_cost_threshold=np.inf,
            walltime_limit=self.total_walltime_limit,
            trial_walltime_limit=self.func_eval_time_limit,
            trial_memory_limit=self.memory_limit,
            n_trials=self.runcount_limit if self.runcount_limit is not None else sys.maxsize,
            instance_features=None,
            min_budget=self.initial_budget,
            max_budget=100,
            seed=self.seed,
            n_workers=self.n_jobs
        )

        ta_kwargs = {
            'backend': copy.deepcopy(self.backend),
            'seed': self.seed,
            'random_state': self.random_state,
            'splitter': self.splitter,
            'metric': self.metric,
            'use_pynisher': self.use_pynisher,
            'refit': self.refit,
            'verbose': self.verbose,
        }

        y = self.datamanager.y
        initial_configs = self.get_initial_configs(y)
        priors = self.get_hp_priors(y)

        callbacks = []
        if self.ensemble_callback is not None:
            callbacks.append(self.ensemble_callback)
        if self.trials_callback is not None:
            callbacks.append(self.trials_callback)

        smac = self.intensifier_generator(
            scenario=scenario,
            seed=self.seed,
            ta_kwargs=ta_kwargs,
            initial_configurations=initial_configs,
            callbacks=callbacks,
            priors=priors
        )

        return smac

    def get_initial_configs(self, y: pd.Series) -> Optional[List[Configuration]]:
        initial_configs = [self.config_space.get_default_configuration()]

        if self.num_metalearning_configs < 1:
            return initial_configs
        try:
            metabase = self._get_metabase()
            return initial_configs + metabase.suggest_configs(y, self.num_metalearning_configs)
        except FileNotFoundError:
            self.logger.warning(f'Failed to find metadata in \'{self.metadata_directory}\'. Skipping meta-learning...')
            return initial_configs

    def get_hp_priors(self, y: pd.Series) -> Optional[Dict[str, Prior]]:
        if not self.hp_priors:
            return None
        try:
            metabase = self._get_metabase()
            priors = metabase.suggest_univariate_prior(y, self.num_metalearning_configs)
            if len(priors) == 0:
                return None
            return priors
        except FileNotFoundError:
            self.logger.warning(f'Failed to find metadata in \'{self.metadata_directory}\'. Skipping meta-learning...')
            return None

    def _get_metabase(self) -> MetaBase:
        if not os.path.exists(self.metadata_directory):
            raise ValueError(f'The specified metadata directory \'{self.metadata_directory}\' does not exist!')

        metabase = MetaBase(
            configuration_space=self.config_space,
            task=self.datamanager.info['task'],
            metric=METRIC_TO_STRING[type(self.metric)],
            base_dir=self.metadata_directory
        )
        return metabase
