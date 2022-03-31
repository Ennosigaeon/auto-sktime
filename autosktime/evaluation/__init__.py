import functools
import logging
import math
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import pynisher
from ConfigSpace import Configuration
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.tae.execute_func import AbstractTAFunc

from autosktime.automl_common.common.utils.backend import Backend
from autosktime.metrics import Scorer

TaFuncResult = NamedTuple('TaFuncResult', [
    ('loss', float),
    ('status', StatusType),
    ('additional_run_info', Dict[str, Any])
])


def fit_predict_try_except_decorator(
        ta: Callable,
        cost_for_crash: float,
        **kwargs: Any) -> TaFuncResult:
    try:
        return ta(**kwargs)
    except Exception as e:
        if isinstance(e, (MemoryError, pynisher.TimeoutException)):
            # Re-raise the memory error to let the pynisher handle that correctly
            raise e

        exception_traceback = traceback.format_exc()
        error_message = repr(e)

        print("Exception handling in `fit_predict_try_except_decorator`: "
              "traceback: {} \nerror message: {}".format(exception_traceback, error_message))

        return TaFuncResult(
            loss=cost_for_crash,
            status=StatusType.CRASHED,
            additional_run_info={'traceback': exception_traceback, 'error': error_message}
        )


def get_cost_of_crash(metric: Scorer) -> float:
    # This function translates worst_possible_result to be a minimization problem. For metrics like accuracy that are
    # bounded to [0,1] metric.optimum == 1 is the worst cost.
    if metric._sign > 0:
        worst_possible_result = metric._worst_possible_result
    else:
        worst_possible_result = metric.optimum - metric._worst_possible_result

    return worst_possible_result


class ExecuteTaFunc(AbstractTAFunc):

    def __init__(
            self,
            backend: Backend,
            seed: int,
            resampling_strategy: str,
            metric: Scorer,
            stats: Stats,
            memory_limit: Optional[int] = None,
            budget_type: Optional[str] = None,
            use_pynisher: bool = True,
            resampling_strategy_args: Dict[str, Any] = None,
            ta: Optional[Callable] = None,
            **kwargs
    ):
        if resampling_strategy == 'holdout':
            from autosktime.evaluation.train_evaluator import eval_holdout
            eval_function = eval_holdout
        else:
            raise ValueError('Unknown resampling strategy %s' %
                             resampling_strategy)

        self.worst_possible_result = get_cost_of_crash(metric)

        eval_function = functools.partial(
            fit_predict_try_except_decorator,
            ta=eval_function,
            cost_for_crash=self.worst_possible_result,
        )

        super().__init__(
            ta=eval_function,
            stats=stats,
            use_pynisher=use_pynisher,
            **kwargs
        )

        self.backend = backend
        self.seed = seed
        self.resampling_strategy = resampling_strategy
        self.metric = metric
        self.resampling_strategy = resampling_strategy
        self.budget_type = budget_type

        if resampling_strategy_args is None:
            resampling_strategy_args = {}
        self.resampling_strategy_args = resampling_strategy_args

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit

        self.logger: logging.Logger = logging.getLogger("TAE")

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
        if self.budget_type is None:
            if run_info.budget != 0:
                raise ValueError('If budget_type is None, budget must be.0, but is {}'.format(run_info.budget))
        else:
            raise NotImplementedError('budgets not supported yet')

        remaining_time = self.stats.get_remaing_time_budget()

        if remaining_time - 5 < run_info.cutoff:
            run_info = run_info._replace(cutoff=int(remaining_time - 5))

        if run_info.cutoff < 1.0:
            self.logger.info("Not starting configuration {} because time is up".format(run_info.config.config_id))
            return run_info, RunValue(
                status=StatusType.STOP,
                cost=self.worst_possible_result,
                time=0.0,
                additional_info={},
                starttime=time.time(),
                endtime=time.time(),
            )
        elif run_info.cutoff != int(np.ceil(run_info.cutoff)) and not isinstance(run_info.cutoff, int):
            run_info = run_info._replace(cutoff=int(np.ceil(run_info.cutoff)))

        self.logger.info("Starting to evaluate configuration {}".format(run_info.config.config_id))
        info, value = super().run_wrapper(run_info=run_info)

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

        return info, value

    def _call_ta(
            self,
            obj: Callable,
            config: Configuration,
            obj_kwargs: Dict[str, Union[int, str, float, None]],
            instance: Optional[str] = None,
            cutoff: Optional[float] = None,
            seed: int = 12345,
            budget: float = 0.0,
            instance_specific: Optional[str] = None,
    ) -> Tuple[float, Dict[str, Union[int, float, str, Dict, List, Tuple]]]:
        info: TaFuncResult = obj(
            config=config,
            backend=self.backend,
            metric=self.metric,
            seed=self.seed,
            num_run=config.config_id,
            budget=budget,
            budget_type=self.budget_type
        )

        cost = info.loss
        additional_run_info = info.additional_run_info
        additional_run_info['configuration_origin'] = config.origin
        additional_run_info['status'] = info.status

        self.logger.info("Finished evaluating configuration {}".format(config.config_id))
        return cost, additional_run_info
