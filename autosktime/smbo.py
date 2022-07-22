import copy
import logging
import os
from typing import Optional, Tuple, List, Dict, Type, Any

import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

import autosktime
from ConfigSpace import ConfigurationSpace, Configuration
from autosktime.automl_common.common.utils.backend import Backend
from autosktime.data import DataManager
from autosktime.data.splitter import BaseSplitter
from autosktime.evaluation import ExecuteTaFunc, get_cost_of_crash
from autosktime.metalearning.meta_base import MetaBase
from autosktime.metrics import METRIC_TO_STRING
from autosktime.smac.prior import Prior
from smac.callbacks import IncorporateRunResultCallback
from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.intensification.successive_halving import SuccessiveHalving
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajEntry


class IntensifierGenerator:

    def __call__(self, *args, **kwargs) -> SMAC4AC:
        pass


class SimpleIntensifierGenerator(IntensifierGenerator):

    def __call__(
            self,
            scenario: Scenario,
            seed: int,
            ta_kwargs: Dict,
            initial_configurations: List[Configuration],
            hp_priors: bool,
            priors: Dict[str, Prior]
    ) -> SMAC4AC:
        return SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=RunHistory2EPM4LogCost,
            tae_runner=ExecuteTaFunc,
            tae_runner_kwargs=ta_kwargs,
            run_id=seed,
            intensifier=SimpleIntensifier,
            initial_configurations=initial_configurations,
            user_priors=hp_priors,
            user_prior_kwargs={
                'decay_beta': 10,
                'priors': priors
            },
        )


class SHIntensifierGenerator(IntensifierGenerator):

    # TODO reset eta and initial_budget to (4.0, 5.0)
    def __init__(self, budget_type: str = 'iterations', eta: float = 2.0, initial_budget: float = 50.0):
        self.budget_type = budget_type
        self.eta = eta
        self.initial_budget = initial_budget

    def __call__(
            self,
            scenario: Scenario,
            seed: int,
            ta_kwargs: Dict,
            initial_configurations: List[Configuration],
            hp_priors: bool,
            priors: Dict[str, Prior]
    ) -> SMAC4AC:
        ta_kwargs['budget_type'] = self.budget_type

        return SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=RunHistory2EPM4LogCost,
            tae_runner=ExecuteTaFunc,
            tae_runner_kwargs=ta_kwargs,
            run_id=seed,
            intensifier=SuccessiveHalving,
            intensifier_kwargs={
                'initial_budget': self.initial_budget,
                'max_budget': 100,
                'eta': self.eta,
                'min_chall': 1,
            },
            initial_configurations=initial_configurations,
            user_priors=hp_priors,
            user_prior_kwargs={
                'decay_beta': 10,
                'priors': priors
            },
        )


class AutoMLSMBO:

    def __init__(
            self,
            config_space: ConfigurationSpace,
            datamanager: DataManager,
            backend: Backend,
            total_walltime_limit: float,
            func_eval_time_limit: float,
            memory_limit: float,
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
            ensemble_callback: Optional[IncorporateRunResultCallback] = None,
            trials_callback: Optional[IncorporateRunResultCallback] = None
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

        # Evaluation
        self.splitter = splitter
        self.use_pynisher = use_pynisher

        # and a bunch of useful limits
        self.worst_possible_result = get_cost_of_crash(self.metric)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
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

    def optimize(self) -> Tuple[RunHistory, List[TrajEntry]]:
        if self.num_metalearning_configs > 0 and self.hp_priors:
            raise ValueError(f"'num_metalearning_configs' ({self.num_metalearning_configs}) and "
                             f"'hp_priors' {self.hp_priors} both set.")

        self.config_space.seed(self.seed)

        smac = self._get_smac()

        # Main optimization loop
        smac.optimize()

        self.runhistory = smac.solver.runhistory
        self.trajectory = smac.solver.intensifier.traj_logger.trajectory

        return self.runhistory, self.trajectory

    def _get_smac(self):
        scenario = Scenario({
            'abort_on_first_run_crash': False,
            'save-results-instantly': True,
            'cs': self.config_space,
            'cutoff_time': self.func_eval_time_limit,
            'deterministic': 'true',
            'memory_limit': self.memory_limit,
            'output-dir': self.backend.get_smac_output_directory(),
            'run_obj': 'quality',
            'wallclock_limit': self.total_walltime_limit,
            'cost_for_crash': self.worst_possible_result,
            'intens_min_chall': 1,
        })

        ta_kwargs = {
            'backend': copy.deepcopy(self.backend),
            'seed': self.seed,
            'random_state': self.random_state,
            'splitter': self.splitter,
            'metric': self.metric,
            'memory_limit': self.memory_limit,
            'use_pynisher': self.use_pynisher,
            'debug_log': self.verbose,
        }

        y = self.datamanager.y
        initial_configs = self.get_initial_configs(y)
        priors = self.get_hp_priors(y)

        smac = self.intensifier_generator(scenario, self.seed, ta_kwargs, initial_configs, self.hp_priors, priors)

        if self.ensemble_callback is not None:
            smac.register_callback(self.ensemble_callback)
        if self.trials_callback is not None:
            smac.register_callback(self.trials_callback)

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
