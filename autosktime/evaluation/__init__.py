import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pynisher
from ConfigSpace import Configuration
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric
from smac import Scenario
from smac.runhistory import StatusType
from smac.runhistory.runhistory import TrialInfo, TrialValue
from smac.runner import TargetFunctionRunner

from autosktime.automl_common.common.utils.backend import Backend
from autosktime.constants import Budget
from autosktime.data.splitter import BaseSplitter
from autosktime.pipeline.templates import TemplateChoice
from autosktime.util.backend import ConfigContext
from autosktime.util.context import Restorer

TaFuncResult = Tuple[float, Dict[str, Any]]


def fit_predict_try_except_decorator(
        config: Configuration,
        context: 'ExecuteTaFunc',
        budget: float = 0.0,
        **kwargs: Any) -> TaFuncResult:
    try:
        return context.eval_function(
            config=config,
            budget=budget,
            splitter=context.splitter,
            random_state=context.random_state,
            budget_type=context.budget_type,
            verbose=context.verbose,
            num_run=config.config_id,
            backend=context.backend,
            metric=context.metric,
            seed=context.seed,
            refit=context.refit,
            **kwargs
        )
    except Exception as e:
        if isinstance(e, (MemoryError, pynisher.TimeoutException)):
            # Re-raise the memory error to let the pynisher handle that correctly
            raise

        logging.getLogger('TAE').exception('Exception handling in `fit_predict_try_except_decorator`:')
        raise


class ExecuteTaFunc(TargetFunctionRunner):

    def __init__(
            self,
            scenario: Scenario,
            backend: Backend,
            seed: int,
            random_state: np.random.RandomState,
            splitter: Optional[BaseSplitter],
            metric: BaseForecastingErrorMetric,
            budget_type: Optional[Budget] = None,
            refit: bool = False,
            use_pynisher: bool = True,
            ta: Optional[Callable] = None,
            verbose: bool = False
    ):
        super().__init__(
            scenario=scenario,
            target_function=fit_predict_try_except_decorator,
            required_arguments=['budget', 'kwargs']
        )
        self.backend = backend
        self.seed = seed
        self.random_state = random_state
        self.splitter = splitter
        self.metric = metric
        self.budget_type = budget_type
        self.refit = refit
        self.use_pynisher = use_pynisher
        self.verbose = verbose
        self.num_run = 0

        if ta is None:
            from autosktime.evaluation.train_evaluator import evaluate
            self.eval_function = evaluate
        else:
            self.eval_function = ta

        self.dataset_properties = self.backend.load_datamanager().dataset_properties
        self.logger: logging.Logger = logging.getLogger('TAE')

    def __call__(
            self,
            config: Configuration,
            algorithm: Callable,
            algorithm_kwargs: dict[str, Any],
    ) -> (
            float
            | list[float]
            | dict[str, float]
            | tuple[float, dict]
            | tuple[list[float], dict]
            | tuple[dict[str, float], dict]
    ):
        """Calls the algorithm, which is processed in the ``run`` method."""
        return algorithm(config, context=self, **algorithm_kwargs)

    def run_wrapper(
            self,
            run_info: TrialInfo,
    ) -> Tuple[TrialInfo, TrialValue]:
        """
        wrapper function for ExecuteTARun.run_wrapper() to cap the target algorithm
        runtime if it would run over the total allowed runtime.

        Parameters
        ----------
        run_info : RunInfo
            Object that contains enough information to execute a configuration run in
            isolation.
        Returns
        -------
        TrialInfo:
            an object containing the configuration launched
        TrialValue:
            Contains information about the status/performance of config
        """
        if self.budget_type is None and run_info.budget:
            raise ValueError(f'If budget_type is None, budget must not be given, but is {run_info.budget}')

        if run_info.config.config_id is None:
            run_info.config.config_id = self.num_run
        self.num_run += 1

        # Provide additional configuration for model evaluation
        config_context: ConfigContext = ConfigContext.instance()
        config_context.set_config(run_info.config.config_id, {
            'start': time.time(),
            'budget': run_info.budget
        })

        self.logger.info(
            f'Starting to evaluate configuration {run_info.config.config_id} with budget {run_info.budget}: {run_info.config.get_dictionary()}')

        with Restorer(self, '_memory_limit', '_algorithm_walltime_limit'):
            if not self._use_pynisher(run_info):
                self._memory_limit = None
                self._algorithm_walltime_limit = None
            info, value = super().run_wrapper(trial_info=run_info)

        config_context.reset_config(run_info.config.config_id)

        if 'status' in value.additional_info:
            # smac treats all ta calls without an exception as a success independent of the provided status
            if value.additional_info['status'] != value.status:
                value = TrialValue(
                    cost=value.cost,
                    time=value.time,
                    status=value.additional_info['status'],
                    starttime=value.starttime,
                    endtime=value.endtime,
                    additional_info=value.additional_info
                )
            del value.additional_info['status']

        self.logger.info(f'Finished evaluating configuration {run_info.config.config_id} with loss {value.cost} and '
                         f'status {value.status}')

        return info, value

    def _use_pynisher(self, run_info: TrialInfo) -> bool:
        if self.use_pynisher:
            # Check if actual model supports pynisher
            try:
                model = TemplateChoice.from_config(run_info.config, run_info.budget, self.dataset_properties)
                return model.supports_pynisher()
            except Exception:
                self.logger.exception('Failed to create model from config. Assuming pynisher is not supported')
                return False
        return self.use_pynisher

    def _call_ta(
            self,
            obj: Callable,
            config: Configuration,
            obj_kwargs: Dict[str, Union[int, str, float, None]],
            cutoff: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Union[int, float, str, Dict, List, Tuple]]]:
        info: TaFuncResult = obj(
            config=config,
            backend=self.backend,
            metric=self.metric,
            num_run=config.config_id,
            budget_type=self.budget_type,
            **obj_kwargs
        )

        if info is None:
            # Execution was not successful
            return 0, {}

        cost, additional_run_info = info
        additional_run_info['configuration_origin'] = config.origin
        additional_run_info['status'] = StatusType.SUCCESS

        return cost, additional_run_info
