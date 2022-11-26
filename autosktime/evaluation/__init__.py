import functools
import logging
import math
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import pynisher
# noinspection PyProtectedMember
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from ConfigSpace import Configuration
from autosktime.automl_common.common.utils.backend import Backend
from autosktime.data.splitter import BaseSplitter
from autosktime.metrics import get_cost_of_crash
from autosktime.pipeline.templates import TemplateChoice
from autosktime.util.backend import ConfigContext
from autosktime.util.context import Restorer
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.tae.execute_func import AbstractTAFunc

TaFuncResult = NamedTuple('TaFuncResult', [
    ('loss', float),
    ('status', StatusType),
    ('additional_run_info', Dict[str, Any])
])


def fit_predict_try_except_decorator(
        ta: Callable,
        cost_for_crash: float,
        seed: int = 0,
        budget: float = 0.0,
        **kwargs: Any) -> TaFuncResult:
    try:
        return ta(seed=seed, budget=budget, **kwargs)
    except Exception as e:
        if isinstance(e, (MemoryError, pynisher.TimeoutException)):
            # Re-raise the memory error to let the pynisher handle that correctly
            raise e

        logging.getLogger('TAE').exception('Exception handling in `fit_predict_try_except_decorator`:')

        return TaFuncResult(
            loss=cost_for_crash,
            status=StatusType.CRASHED,
            additional_run_info={
                'traceback': traceback.format_exc(),
                'error': repr(e)
            }
        )


class ExecuteTaFunc(AbstractTAFunc):

    def __init__(
            self,
            backend: Backend,
            seed: int,
            random_state: np.random.RandomState,
            splitter: Optional[BaseSplitter],
            metric: BaseForecastingErrorMetric,
            stats: Stats,
            memory_limit: Optional[int] = None,
            budget_type: Optional[str] = None,
            use_pynisher: bool = True,
            ta: Optional[Callable] = None,
            verbose: bool = False,
            **kwargs
    ):
        if ta is None:
            from autosktime.evaluation.train_evaluator import evaluate
            eval_function = evaluate
        else:
            eval_function = ta

        self.worst_possible_result = get_cost_of_crash(metric)

        eval_function = functools.partial(
            fit_predict_try_except_decorator,
            ta=eval_function,
            cost_for_crash=self.worst_possible_result,
            splitter=splitter,
            random_state=random_state,
            budget_type=budget_type,
            verbose=verbose,
        )

        super().__init__(
            ta=eval_function,
            stats=stats,
            use_pynisher=use_pynisher,
            **kwargs
        )

        self.backend = backend
        self.seed = seed
        self.metric = metric
        self.budget_type = budget_type

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit

        self.dataset_properties = self.backend.load_datamanager().dataset_properties
        self.logger: logging.Logger = logging.getLogger('TAE')

    def run_wrapper(
            self,
            run_info: RunInfo,
    ) -> Tuple[RunInfo, RunValue]:
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
        RunInfo:
            an object containing the configuration launched
        RunValue:
            Contains information about the status/performance of config
        """
        if self.budget_type is None and run_info.budget != 0:
            raise ValueError(f'If budget_type is None, budget must be 0.0, but is {run_info.budget}')

        remaining_time = self.stats.get_remaing_time_budget()

        if remaining_time - 5 < run_info.cutoff:
            # noinspection PyProtectedMember
            run_info = run_info._replace(cutoff=int(remaining_time - 5))

        if run_info.cutoff < 1.0:
            self.logger.info(f'Not starting configuration {run_info.config.config_id} because time is up')
            return run_info, RunValue(
                status=StatusType.STOP,
                cost=self.worst_possible_result,
                time=0.0,
                additional_info={},
                starttime=time.time(),
                endtime=time.time(),
            )
        elif run_info.cutoff != int(np.ceil(run_info.cutoff)) and not isinstance(run_info.cutoff, int):
            # noinspection PyProtectedMember
            run_info = run_info._replace(cutoff=int(np.ceil(run_info.cutoff)))

        # Provide additional configuration for model evaluation
        config_context: ConfigContext = ConfigContext.instance()
        config_context.set_config(run_info.config.config_id, {
            'start': time.time(),
            'cutoff': run_info.cutoff,
            'budget': run_info.budget
        })

        self.logger.info(
            f'Starting to evaluate configuration {run_info.config.config_id}: {run_info.config.get_dictionary()}')

        with Restorer(self, 'use_pynisher'):
            self.use_pynisher = self._use_pynisher(run_info)
            info, value = super().run_wrapper(run_info=run_info)

        config_context.reset_config(run_info.config.config_id)

        if 'status' in value.additional_info:
            # smac treats all ta calls without an exception as a success independent of the provided status
            if value.additional_info['status'] != value.status:
                value = RunValue(
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

    def _use_pynisher(self, run_info: RunInfo) -> bool:
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

        cost = info.loss
        additional_run_info = info.additional_run_info
        additional_run_info['configuration_origin'] = config.origin
        additional_run_info['status'] = info.status

        return cost, additional_run_info
