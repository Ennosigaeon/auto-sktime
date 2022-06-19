import copy
import logging
import os
from typing import Optional, Tuple, List, Dict

import autosktime
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from autosktime.automl_common.common.utils.backend import Backend
from autosktime.data import AbstractDataManager
from autosktime.data.splitter import BaseSplitter
from autosktime.evaluation import ExecuteTaFunc, get_cost_of_crash
from autosktime.metalearning.acquisition import PriorAcquisitionFunction
from autosktime.metalearning.meta_base import MetaBase
from autosktime.metalearning.prior import Prior
from autosktime.metrics import BaseMetric, METRIC_TO_STRING

from smac.callbacks import IncorporateRunResultCallback
from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajEntry


class AutoMLSMBO:

    def __init__(
            self,
            config_space: ConfigurationSpace,
            datamanager: AbstractDataManager,
            backend: Backend,
            total_walltime_limit: float,
            func_eval_time_limit: float,
            memory_limit: float,
            metric: BaseMetric,
            splitter: BaseSplitter,
            use_pynisher: bool = True,
            seed: int = 1,
            metadata_directory: str = None,
            num_metalearning_configs: int = -1,
            hp_priors: bool = False,
            ensemble_callback: Optional[IncorporateRunResultCallback] = None,
            trials_callback: Optional[IncorporateRunResultCallback] = None
    ):
        # data related
        self.datamanager = datamanager
        self.metric = metric
        self.backend = backend

        # the configuration space
        self.config_space = config_space

        # Evaluation
        self.splitter = splitter
        self.use_pynisher = use_pynisher

        # and a bunch of useful limits
        self.worst_possible_result = get_cost_of_crash(self.metric)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.memory_limit = memory_limit
        self.seed = seed

        # metalearning
        self.metadata_directory = metadata_directory
        self.num_metalearning_configs = num_metalearning_configs
        self.hp_priors = hp_priors

        self.ensemble_callback = ensemble_callback
        self.trials_callback = trials_callback

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
        })

        ta_kwargs = {
            'backend': copy.deepcopy(self.backend),
            'seed': self.seed,
            'splitter': self.splitter,
            'metric': self.metric,
            'memory_limit': self.memory_limit,
            'use_pynisher': self.use_pynisher,
        }

        y = self.datamanager.y
        initial_configurations = self.get_initial_configs(y)
        priors = self.get_hp_priors(y)

        smac = SMAC4AC(
            scenario=scenario,
            rng=self.seed,
            runhistory2epm=RunHistory2EPM4LogCost,
            tae_runner=ExecuteTaFunc,
            tae_runner_kwargs=ta_kwargs,
            run_id=self.seed,
            intensifier=SimpleIntensifier,
            initial_configurations=initial_configurations,
            user_priors=self.hp_priors,
            user_prior_instance=PriorAcquisitionFunction,
            user_prior_kwargs={
                'decay_beta': 10,
                'priors': priors
            },
        )

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
