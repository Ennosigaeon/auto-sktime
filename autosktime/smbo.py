import copy
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Type, Any

import dask.distributed
import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

import autosktime
from ConfigSpace import ConfigurationSpace, Configuration
from autosktime.automl_common.common.utils.backend import Backend
from autosktime.data import DataManager
from autosktime.data.splitter import BaseSplitter
from autosktime.evaluation import ExecuteTaFunc
from autosktime.metalearning.meta_base import MetaBase
from autosktime.metrics import METRIC_TO_STRING, get_cost_of_crash
from autosktime.smac.acquisition import PriorAcquisitionFunction
from autosktime.smac.prior import Prior
from smac import Callback, AlgorithmConfigurationFacade, MultiFidelityFacade
from smac.facade import AbstractFacade
from smac.initial_design import DefaultInitialDesign
from smac.intensifier import SuccessiveHalving
from smac.intensifier.intensifier import Intensifier
from smac.runhistory.dataclasses import TrajectoryItem
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


class IntensifierGenerator:

    def __call__(self, *args, **kwargs) -> AbstractFacade:
        pass


class SimpleIntensifierGenerator(IntensifierGenerator):

    def __call__(
            self,
            scenario: Scenario,
            ta_kwargs: Dict,
            initial_configurations: List[Configuration],
            hp_priors: bool,
            priors: Dict[str, Prior],
            callbacks: List[Callback]
    ) -> AbstractFacade:
        ta_kwargs['scenario'] = scenario

        initial_design = None
        if initial_configurations is not None and len(initial_configurations) > 0:
            initial_design = DefaultInitialDesign(
                scenario=scenario,
                n_configs=0,
                additional_configs=initial_configurations,
            )

        acquisition_function = AlgorithmConfigurationFacade.get_acquisition_function(scenario)
        if hp_priors:
            acquisition_function = PriorAcquisitionFunction(acquisition_function, decay_beta=10, priors=priors)

        return AlgorithmConfigurationFacade(
            scenario=scenario,
            target_function=ExecuteTaFunc(**ta_kwargs),
            intensifier=Intensifier(scenario),
            initial_design=initial_design,
            callbacks=callbacks,
            acquisition_function=acquisition_function
        )


class SHIntensifierGenerator(IntensifierGenerator):

    # Use eta and initial_budget as (2.0, 50.0) for 2 candidates per iteration with 50% elimination
    def __init__(self, budget_type: str = 'iterations', eta: int = 4.0):
        self.budget_type = budget_type
        self.eta = eta

    def __call__(
            self,
            scenario: Scenario,
            seed: int,
            ta_kwargs: Dict,
            initial_configurations: List[Configuration],
            hp_priors: bool,
            priors: Dict[str, Prior],
            callbacks: List[Callback]
    ) -> AbstractFacade:
        ta_kwargs['scenario'] = scenario
        ta_kwargs['budget_type'] = self.budget_type

        initial_design = None
        if initial_configurations is not None and len(initial_configurations) > 0:
            initial_design = DefaultInitialDesign(
                scenario=scenario,
                n_configs=0,
                additional_configs=initial_configurations,
            )

        acquisition_function = MultiFidelityFacade.get_acquisition_function(scenario)
        if hp_priors:
            acquisition_function = PriorAcquisitionFunction(acquisition_function, decay_beta=10, priors=priors)

        return MultiFidelityFacade(
            scenario=scenario,
            target_function=ExecuteTaFunc(**ta_kwargs),
            intensifier=SuccessiveHalving(scenario, eta=self.eta, seed=seed),
            initial_design=initial_design,
            callbacks=callbacks,
            acquisition_function=acquisition_function
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

        if self.num_metalearning_configs > 0 and self.hp_priors:
            raise ValueError(f"'num_metalearning_configs' ({self.num_metalearning_configs}) and "
                             f"'hp_priors' {self.hp_priors} both set.")

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
            min_budget=initial_budget,
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
            hp_priors=self.hp_priors,
            priors=priors
        )

        return smac

    def get_initial_configs(self, y: pd.Series) -> Optional[List[Configuration]]:
        if self.num_metalearning_configs < 0:
            return None
        try:
            metabase = self._get_metabase()
            return metabase.suggest_configs(y, self.num_metalearning_configs)
        except FileNotFoundError:
            self.logger.warning(f'Failed to find metadata in \'{self.metadata_directory}\'. Skipping meta-learning...')
            return None

    def get_hp_priors(self, y: pd.Series) -> Optional[Dict[str, Prior]]:
        if not self.hp_priors:
            return None
        try:
            metabase = self._get_metabase()
            return metabase.suggest_univariate_prior(y, self.num_metalearning_configs)
        except FileNotFoundError:
            self.logger.warning(f'Failed to find metadata in \'{self.metadata_directory}\'. Skipping meta-learning...')
            return None

    def _get_metabase(self) -> MetaBase:
        if self.metadata_directory is None:
            metalearning_directory = os.path.dirname(autosktime.metalearning.__file__)
            metadata_directory = os.path.join(metalearning_directory, 'files')
            self.metadata_directory = metadata_directory

        if not os.path.exists(self.metadata_directory):
            raise ValueError(f'The specified metadata directory \'{self.metadata_directory}\' does not exist!')

        metabase = MetaBase(
            configuration_space=self.config_space,
            task=self.datamanager.info['task'],
            metric=METRIC_TO_STRING[type(self.metric)],
            base_dir=self.metadata_directory
        )
        return metabase
