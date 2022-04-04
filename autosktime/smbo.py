import copy
import logging
from typing import Dict, Any, Optional

from smac.callbacks import IncorporateRunResultCallback
from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario

from ConfigSpace import ConfigurationSpace
from autosktime.automl_common.common.utils.backend import Backend
from autosktime.data import AbstractDataManager
from autosktime.evaluation import ExecuteTaFunc, get_cost_of_crash
from autosktime.metrics import Scorer


class AutoMLSMBO:

    def __init__(
            self,
            config_space: ConfigurationSpace,
            datamanager: AbstractDataManager,
            backend: Backend,
            total_walltime_limit: float,
            func_eval_time_limit: float,
            memory_limit: float,
            metric: Scorer,
            seed: int = 1,
            resampling_strategy: str = 'holdout',
            resampling_strategy_args: Dict[str, Any] = None,
            trials_callback: Optional[IncorporateRunResultCallback] = None
    ):
        # data related
        self.datamanager = datamanager
        self.metric = metric
        self.backend = backend

        # the configuration space
        self.config_space = config_space

        # Evaluation
        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            resampling_strategy_args = {}
        self.resampling_strategy_args = resampling_strategy_args

        # and a bunch of useful limits
        self.worst_possible_result = get_cost_of_crash(self.metric)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.memory_limit = memory_limit
        self.seed = seed

        self.trials_callback = trials_callback

        self.logger = logging.getLogger(__name__)

    def optimize(self):
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
            'instances': [self.datamanager.name],
            'memory_limit': self.memory_limit,
            'output-dir': self.backend.get_smac_output_directory(),
            'run_obj': 'quality',
            'wallclock_limit': self.total_walltime_limit,
            'cost_for_crash': self.worst_possible_result,
        })

        ta_kwargs = {
            'backend': copy.deepcopy(self.backend),
            'seed': self.seed,
            'resampling_strategy': self.resampling_strategy,
            'resampling_strategy_args': self.resampling_strategy_args,
            'metric': self.metric,
            'memory_limit': self.memory_limit,
        }

        smac = SMAC4AC(
            scenario=scenario,
            rng=self.seed,
            runhistory2epm=RunHistory2EPM4LogCost,
            tae_runner=ExecuteTaFunc,
            tae_runner_kwargs=ta_kwargs,
            run_id=self.seed,
            intensifier=SimpleIntensifier,
        )

        if self.trials_callback is not None:
            smac.register_callback(self.trials_callback)

        return smac
