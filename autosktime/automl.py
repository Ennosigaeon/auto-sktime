import copy
import logging.handlers
import os
import platform
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, Optional, List, Tuple, Union

import dask
import dask.distributed
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.read_and_write import json as cs_json
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric
from smac import Scenario
from smac.runhistory.enumerations import StatusType
from smac.runhistory.runhistory import TrialInfo

import autosktime.evaluation.test_evaluator
from autosktime.automl_common.common.ensemble_building.abstract_ensemble import AbstractEnsemble
from autosktime.constants import SUPPORTED_Y_TYPES, PANEL_FORECAST, MULTIVARIATE_FORECAST, UNIVARIATE_FORECAST, \
    Budget
from autosktime.data import DataManager, DatasetProperties
from autosktime.data.splitter import BaseSplitter, splitter_types, get_ensemble_data
from autosktime.ensembles.builder import EnsembleBuilderManager
from autosktime.ensembles.singlebest import SingleBest
from autosktime.ensembles.util import PrefittedEnsembleForecaster
from autosktime.evaluation import ExecuteTaFunc
from autosktime.metrics import default_metric_for_task
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES
from autosktime.pipeline.templates import util, TemplateChoice
from autosktime.pipeline.util import NotVectorizedMixin
from autosktime.smbo import AutoMLSMBO, SHIntensifierGenerator, SimpleIntensifierGenerator
from autosktime.util.backend import create, Backend
from autosktime.util.logging_ import setup_logger
from autosktime.util.stopwatch import StopWatch


class AutoML(NotVectorizedMixin, AutoSktimePredictor):
    _tags = {
        'requires-fh-in-fit': False,
        'X-y-must-have-same-index': True,
        'capability:pred_int': True
    }

    def __init__(self,
                 time_left_for_this_task: int,
                 per_run_time_limit: int,
                 runcount_limit: int = None,
                 working_directory: Optional[str] = None,
                 delete_tmp_folder_after_terminate: bool = True,
                 ensemble_size: int = 1,
                 ensemble_nbest: int = 1,
                 max_models_on_disc: int = 1,
                 seed: int = None,
                 memory_limit: int = None,
                 include: Optional[Dict[str, Optional[List[str]]]] = None,
                 exclude: Optional[Dict[str, Optional[List[str]]]] = None,
                 resampling_strategy: str = 'temporal-holdout',
                 resampling_strategy_arguments: Dict[str, Any] = None,
                 metadata_directory: str = None,
                 num_metalearning_configs: int = -1,
                 hp_priors: bool = False,
                 n_jobs: int = 1,
                 dask_client: Optional[dask.distributed.Client] = None,
                 logging_config: Dict[str, Any] = None,
                 metric: BaseForecastingErrorMetric = None,
                 use_pynisher: bool = False,
                 budget: Optional[Budget] = None,
                 refit: bool = False,
                 verbose: bool = False
                 ):
        super(AutoML, self).__init__()
        self.configuration_space: Optional[ConfigurationSpace] = None
        self._backend: Optional[Backend] = None
        self._working_directory = working_directory
        self._delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self._per_run_time_limit = per_run_time_limit
        self._ensemble_size = ensemble_size
        self._ensemble_nbest = ensemble_nbest
        self._max_models_on_disc = max_models_on_disc
        self._memory_limit = memory_limit
        self._include = include
        self._exclude = exclude
        self._resampling_strategy = resampling_strategy
        res_strat = resampling_strategy_arguments if resampling_strategy_arguments is not None else {}
        self._resampling_strategy_arguments = res_strat
        self._metadata_directory = metadata_directory
        self._num_metalearning_configs = num_metalearning_configs
        self._hp_priors = hp_priors
        self._n_jobs: int = n_jobs
        self._dask_client: Optional[dask.distributed.Client] = dask_client

        self._random_state = np.random.RandomState(seed)
        if seed is None:
            seed = 1
        self._seed = seed

        self.logging_config: Dict[str, Any] = logging_config
        self._logger: Optional[logging.Logger] = None

        self._metric = metric
        self._use_pynisher = use_pynisher
        self._budget = budget
        self._verbose = verbose
        self._refit = refit

        self._datamanager: Optional[DataManager] = None
        self._dataset_name: Optional[str] = None
        self._stopwatch = StopWatch()
        self._task = None
        self._test_data: Tuple[Optional[pd.Series], Optional[pd.DataFrame]] = (None, None)

        self.models_: List[Tuple[float, TemplateChoice]] = []
        self.ensemble_: Optional[BaseForecaster] = None

        self._time_for_task = time_left_for_this_task
        if not isinstance(self._time_for_task, int):
            raise ValueError(f'time_left_for_this_task not of type integer, but {type(self._time_for_task)}')
        if not isinstance(self._per_run_time_limit, int):
            raise ValueError(f'per_run_time_limit not of type integer, but {type(self._per_run_time_limit)}')
        self._runcount_limit = runcount_limit
        if self._runcount_limit is not None and not isinstance(self._runcount_limit, int):
            raise ValueError(f'runcount_limit not of type integer, but {type(self._runcount_limit)}')

        # Tracks how many runs have been launched. It can be seen as an identifier for each configuration saved to disk
        self.num_run_: int = 0

    def _get_logger(self, name: str) -> logging.Logger:
        setup_logger(
            filename=f'AutoML({self._seed}):{name}.log',
            logging_config=self.logging_config,
            output_dir=self._backend.output_directory,
        )

        return logging.getLogger('AutoML')

    @classmethod
    def _task_type_id(cls, task_type: str) -> int:
        raise NotImplementedError

    @classmethod
    def _supports_task_type(cls, task_type: str) -> bool:
        raise NotImplementedError

    def get_fitted_params(self, deep=True):
        raise NotImplementedError

    def reset(self):
        self.models_ = None
        self.ensemble_ = None

    def fit(
            self,
            y: SUPPORTED_Y_TYPES,
            X: pd.DataFrame = None,
            fh: ForecastingHorizon = None,
            task: Optional[int] = None,
            dataset_name: Optional[str] = None,
            y_test: Optional[SUPPORTED_Y_TYPES] = None,
            X_test: Optional[pd.DataFrame] = None,
            configs: List[Tuple[float, float, Union[Configuration, Dict[str, Union[str, float, int]]]]] = None
    ):
        if isinstance(y.index, pd.MultiIndex):
            assert len(y.index.levels) == 2, 'auto-sktime currently only supports at most two different index levels'

        if dataset_name is None:
            dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))
        self._dataset_name = dataset_name

        if task is None:
            if isinstance(y.index, pd.MultiIndex):
                task = PANEL_FORECAST
            elif isinstance(y, pd.DataFrame):
                task = MULTIVARIATE_FORECAST
            else:
                task = UNIVARIATE_FORECAST
            # By convention exogenous task is endogenous + 1
            task += X is not None
        self._task = task

        self._test_data = (y_test, X_test)

        self._metric = self._determine_metric()

        # Create the backend
        self._backend = self._create_backend()
        self._backend.save_start_time(str(self._seed))
        # noinspection PyProtectedMember
        self._backend._make_internals_directory()

        self._logger = self._get_logger(self._dataset_name)

        # sktime does not allow to pass additional parameter to fit(), use self to store them instead
        self.__configs = configs
        return super().fit(y, X, fh)

    def _fit(self, y: SUPPORTED_Y_TYPES, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        # Check if specific configs should be fitted instead of searching for optimal configuration
        if self.__configs is not None:
            res = self._fit_configs(self.__configs, y, X)
            del self.__configs
            return res

        # If no dask client was provided, we create one, so that we can start an ensemble process in parallel to SMBO
        if self._dask_client is None:
            self._dask_client = self._create_dask_client()
            self._is_dask_client_internally_created = True
        else:
            self._is_dask_client_internally_created = False

        # Produce debug information to the logfile
        self._logger.debug('Starting to print environment information')
        self._logger.debug('  Python version: %s', sys.version.split('\n'))
        self._logger.debug('  System: %s', platform.system())
        self._logger.debug('  Machine: %s', platform.machine())
        self._logger.debug('  Platform: %s', platform.platform())
        self._logger.debug('  Version: %s', platform.version())
        self._logger.debug('  Mac version: %s', platform.mac_ver())
        self._logger.debug('Done printing environment information')

        # Prepare training and ensemble data
        self._stopwatch.start_task(self._dataset_name)
        splitter = self._determine_resampling()
        y_ens, X_ens = get_ensemble_data(y, X, splitter)
        self._backend.save_targets_ensemble(y_ens)
        self._backend.save_targets_test(self._test_data[0])
        self._datamanager = DataManager(self._task, y, X, y_ens, X_ens, self._test_data[0], self._test_data[1],
                                        self._dataset_name, splitter)
        self._backend.save_datamanager(self._datamanager)
        time_for_load_data = self._stopwatch.wall_elapsed(self._dataset_name)

        time_left = max(0., self._time_for_task - time_for_load_data)
        self._logger.debug(f'Remaining time after reading {self._dataset_name} {time_left:5.2f} sec')

        # Create a search space
        self.configuration_space = self.get_hyperparameter_search_space()

        # Prepare ensemble builder
        elapsed_time = self._stopwatch.wall_elapsed(self._dataset_name)
        time_left_for_ensembles = max(0., self._time_for_task - elapsed_time)
        proc_ensemble = None
        if time_left_for_ensembles <= 0 < self._ensemble_size:
            raise ValueError('Not starting ensemble builder because there is no time left. Try increasing the '
                             'value of time_left_for_this_task.')
        elif self._ensemble_size <= 0:
            self._logger.info('Not starting ensemble builder because ensemble size is <= 0.')
        else:
            self._logger.info(f'Start Ensemble with {time_left_for_ensembles:5.2f}sec time left')

            proc_ensemble = EnsembleBuilderManager(
                start_time=time.time(),
                time_left_for_ensembles=time_left_for_ensembles,
                backend=copy.deepcopy(self._backend),
                dataset_name=self._dataset_name,
                task=self._task,
                metric=self._metric,
                ensemble_size=self._ensemble_size,
                ensemble_nbest=self._ensemble_nbest,
                max_models_on_disc=self._max_models_on_disc,
                seed=self._seed,
                random_state=self._random_state
            )

        # Run SMAC
        smac_task_name = 'runSMAC'
        self._stopwatch.start_task(smac_task_name)
        elapsed_time = self._stopwatch.wall_elapsed(self._dataset_name)
        time_left_for_smac = max(0., self._time_for_task - elapsed_time)

        if time_left_for_smac <= 0:
            self._logger.warning('Not starting SMAC because there is no time left.')
            _proc_smac = None
        else:
            self._logger.info(f'Starting SMAC with {time_left_for_smac:5.2f}sec time left')
            if self._per_run_time_limit is None or self._per_run_time_limit > time_left_for_smac:
                self._logger.warning('Time limit for a single run is higher than total time limit. Capping the limit '
                                     f'for a single run to the total time given to SMAC ({time_left_for_smac})')
                per_run_time_limit = time_left_for_smac
            else:
                per_run_time_limit = self._per_run_time_limit

            # Make sure that at least 2 models are created for the ensemble process
            if time_left_for_smac // per_run_time_limit < 2:
                per_run_time_limit = time_left_for_smac // 2
                self._logger.warning(f'Capping the per_run_time_limit to {per_run_time_limit} to have time for a least '
                                     f'2 models in each process.')

            _proc_smac = AutoMLSMBO(
                config_space=self.configuration_space,
                datamanager=self._datamanager,
                backend=self._backend,
                total_walltime_limit=time_left_for_smac,
                func_eval_time_limit=per_run_time_limit,
                runcount_limit=self._runcount_limit,
                memory_limit=self._memory_limit,
                n_jobs=self._n_jobs,
                dask_client=self._dask_client,
                metric=self._metric,
                splitter=splitter,
                intensifier_generator=SimpleIntensifierGenerator if self._budget is None else SHIntensifierGenerator,
                intensifier_generator_kwargs={} if self._budget is None else {'budget_type': self._budget},
                initial_budget=25.0 if self._budget == Budget.SeriesLength else 5.0,
                use_pynisher=self._use_pynisher,
                seed=self._seed,
                random_state=self._random_state,
                metadata_directory=self._metadata_directory,
                num_metalearning_configs=self._num_metalearning_configs,
                hp_priors=self._hp_priors,
                refit=False,
                verbose=self._verbose,
                ensemble_callback=proc_ensemble,
            )
            # Reconfigure logger because smac overrides logging configuration
            self._get_logger(self._dataset_name)

            try:
                self.runhistory_, self.trajectory_ = _proc_smac.optimize()
            except Exception as e:
                self._logger.exception(e)
                raise

        self._logger.info('Shutting down...')
        # Wait until the ensemble process is finished to avoid shutting down while it tries to access the data
        if proc_ensemble is not None:
            if proc_ensemble.pending_future is not None:
                # Now we need to wait for the future to return as it cannot be cancelled while it
                # is running: https://stackoverflow.com/a/49203129
                self._logger.info('Ensemble script still running, waiting for it to finish.')
                proc_ensemble.pending_future.result()
                self._logger.info('Ensemble script finished, continue shutdown.')

            # save the ensemble performance history file
            self.ensemble_performance_history_ = proc_ensemble.history
            if len(self.ensemble_performance_history_) > 0:
                pd.DataFrame(self.ensemble_performance_history_,
                             columns=['timestamp', 'performance', 'ensemble_size']).to_json(
                    os.path.join(self._backend.internals_directory, 'ensemble_history.json'))

        self._logger.info('Loading models...')
        self._load_models()
        self._logger.info('Finished loading models...')

        if self._refit:
            self._logger.info('Refitting ensemble on complete data')
            self._fit_configs(self.ensemble_configurations_, y, X, fh)

        self.num_run_ = len(self.runhistory_)
        self._fit_cleanup()

        return self

    def _create_backend(self) -> Backend:
        tmp_dir = self._working_directory + '__tmp__' if self._working_directory is not None else None
        return create(
            temporary_directory=tmp_dir,
            output_directory=self._working_directory,
            prefix='auto-sktime',
            delete_tmp_folder_after_terminate=self._delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=self._working_directory is None
        )

    def _create_dask_client(self) -> Optional[dask.distributed.Client]:
        if self._n_jobs > 1:
            return dask.distributed.Client(
                dask.distributed.LocalCluster(
                    n_workers=self._n_jobs,
                    processes=False,
                    threads_per_worker=1,
                    # Use a tempdir to save the dask workers, because deleting workers takes more time than deleting
                    # backend directories. This prevents an error saying that the worker file was deleted, so the
                    # client could not close the worker properly
                    local_directory=tempfile.gettempdir(),
                    # Memory is handled by the pynisher, not by the dask worker/nanny
                    memory_limit=0,
                ),
                # Heartbeat every 10s
                heartbeat_interval=10000,
            )
        else:
            return None
            # return SingleThreadedClient()

    def _determine_metric(self) -> BaseForecastingErrorMetric:
        metric = self._metric
        if metric is None:
            metric = default_metric_for_task[self._task]
        if metric is None:
            raise ValueError('No metric given.')
        if not isinstance(metric, BaseForecastingErrorMetric):
            raise ValueError(f'Metric must be instance of {type(BaseForecastingErrorMetric)}')
        return metric

    def _determine_resampling(self) -> BaseSplitter:
        # Determine Resampling strategy
        if isinstance(self._resampling_strategy, BaseSplitter):
            splitter = self._resampling_strategy
        elif self._resampling_strategy in splitter_types:
            splitter = splitter_types[self._resampling_strategy](**self._resampling_strategy_arguments)
            # Not all splitters have a random_state argument
            splitter.random_state = self._random_state
        else:
            raise ValueError(f'Unable to create {type(BaseSplitter)} from = {self._resampling_strategy}')
        return splitter

    def _fit_cleanup(self):
        if (
                hasattr(self, '_is_dask_client_internally_created')
                and self._is_dask_client_internally_created
                and self._dask_client
        ):
            self._logger.info('Closing the dask infrastructure')
            self._dask_client.close()
            self._dask_client.shutdown()
            del self._dask_client
            self._dask_client = None
            self._logger.info('Finished closing the dask infrastructure')

        # Clean up the backend
        if self._delete_tmp_folder_after_terminate:
            self._backend.context.delete_directories(force=False)
        return

    def _fit_configs(
            self,
            configs: List[Tuple[float, float, Union[Configuration, Dict[str, Union[str, float, int]]]]],
            y: SUPPORTED_Y_TYPES,
            X: pd.DataFrame = None,
            fh: ForecastingHorizon = None
    ):
        self._logger.info(f'Creating ensemble of {len(configs)} provided configurations. No optimization performed.')

        self._datamanager = DataManager(self._task, y, X, y, X, self._dataset_name)
        self._backend.save_datamanager(self._datamanager)

        # Create a search space
        self.configuration_space = self.get_hyperparameter_search_space()

        self.models_ = []

        for weight, budget, config in configs:
            self.num_run_ += 1
            if isinstance(config, dict):
                config = Configuration(self.configuration_space, config)
            config.config_id = self.num_run_

            # Fit a pipeline, which will be stored on disk which we can later load via the backend
            ta = ExecuteTaFunc(
                scenario=Scenario(self.configuration_space, name='run', seed=self._seed),
                ta=autosktime.evaluation.test_evaluator.evaluate,
                backend=self._backend,
                seed=self._seed,
                random_state=self._random_state,
                splitter=None,
                metric=self._metric,
                use_pynisher=self._use_pynisher,
                refit=False,
                budget_type=Budget.Iterations,
            )

            run_info, run_value = ta.run_wrapper(
                TrialInfo(
                    config=config,
                    instance=None,
                    seed=self._seed,
                    budget=budget,
                )
            )

            if run_value.status != StatusType.SUCCESS:
                raise ValueError(
                    f'Failed to fit configuration {config} with {run_value.status}: {run_value.additional_info}')

            # noinspection PyUnresolvedReferences
            pipeline = self._backend.load_model_by_seed_and_id_and_budget(
                seed=self._seed,
                idx=run_info.config.config_id,
                budget=run_info.budget,
            )
            self.models_.append((weight, pipeline))

        weights, models = tuple(map(list, zip(*self.models_)))
        named_forecasters = [(str(f.config_id), f) for f in models]
        self.ensemble_ = PrefittedEnsembleForecaster(forecasters=named_forecasters, weights=weights)
        self.ensemble_.fit(self._datamanager.y, self._datamanager.X)

        return self

    def check_is_fitted(self):
        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        # If self.ensemble_ is None, it means that ensemble_size is set to zero. In such cases, raise error because
        # predict cannot be called.
        if self.ensemble_ is None:
            raise ValueError('Predict can only be called if ensemble_size != 0')

    def _predict(self, fh: ForecastingHorizon = None, X: pd.DataFrame = None):
        predictions = self.ensemble_.predict(fh=fh, X=X)
        return predictions

    def _predict_interval(self, fh: ForecastingHorizon = None, X: pd.DataFrame = None, coverage: float = 0.90):
        predictions_int = self.ensemble_.predict_interval(fh=fh, X=X)
        return predictions_int

    def _load_models(self) -> None:
        ensemble_ = self._backend.load_ensemble(self._seed)

        # If no ensemble is loaded, try to get the best performing model
        if not ensemble_:
            ensemble_ = self._load_best_individual_model()

        if ensemble_:
            identifiers = ensemble_.get_selected_model_identifiers()
            models_ = self._backend.load_models_by_identifiers(identifiers)

            if len(models_) == 0:
                raise ValueError('No models fitted!')

            # AbstractEnsemble expects string identifiers, but we use PIPELINE_IDENTIFIER
            # noinspection PyTypeChecker
            self.models_: List[Tuple[float, TemplateChoice]] = ensemble_.get_models_with_weights(models_)
            weights, models = tuple(map(list, zip(*self.models_)))
            named_forecasters = [(str(f.config_id), f) for f in models]
            self.ensemble_ = PrefittedEnsembleForecaster(forecasters=named_forecasters, weights=weights)
            self.ensemble_.fit(self._datamanager.y, self._datamanager.X)
        else:
            self.models_ = []

        self.ensemble_configurations_ = []
        for weight, model in self.models_:
            self.ensemble_configurations_.append((weight, model.budget, model.config.get_dictionary()))
        self._logger.info(f'Final weighted ensemble: {self.ensemble_configurations_}')

    def _load_best_individual_model(self) -> Optional[AbstractEnsemble]:
        """
        In case of failure during ensemble building, this method returns the single best model found by AutoML.
        This is a robust mechanism to be able to predict, even though no ensemble was found by ensemble builder.
        """
        # We also require that the model is fit and a task is defined. The ensemble size must also be greater than 1,
        # else it means that the user intentionally does not want an ensemble
        if not self._task or self._ensemble_size < 1:
            return None

        # SingleBest contains the best model found by AutoML
        ensemble = SingleBest(
            metric=self._metric,
            run_history=self.runhistory_,
            seed=self._seed,
            backend=self._backend
        )
        self._logger.warning(
            'No valid ensemble was created. Please check the log file for errors. Default to the best individual '
            'estimator'
        )
        return ensemble

    @property
    def performance_over_time_(self):
        model_perf = self._get_runhistory_models_performance()
        best_values = pd.Series({
            'single_best_optimization_score': -np.inf,
            'single_best_test_score': -np.inf,
            'single_best_train_score': -np.inf
        })
        for idx in model_perf.index:
            if model_perf.loc[idx, 'single_best_optimization_score'] > best_values['single_best_optimization_score']:
                best_values = model_perf.loc[idx]
            model_perf.loc[idx] = best_values

        performance_over_time = model_perf

        if self._ensemble_size != 0:
            ensemble_perf = pd.DataFrame(self.ensemble_performance_history_)
            best_values = pd.Series({'ensemble_optimization_score': -np.inf, 'ensemble_test_score': -np.inf})
            for idx in ensemble_perf.index:
                if ensemble_perf.loc[idx, 'ensemble_optimization_score'] > best_values['ensemble_optimization_score']:
                    best_values = ensemble_perf.loc[idx]
                ensemble_perf.loc[idx] = best_values

            performance_over_time = pd.merge(
                ensemble_perf,
                model_perf,
                on='Timestamp', how='outer'
            ).sort_values('Timestamp').fillna(method='ffill')

        return performance_over_time

    def _get_runhistory_models_performance(self) -> pd.DataFrame:
        performance_list = []
        for run_key, run_value in self.runhistory_.items():
            if run_value.status != StatusType.SUCCESS:
                continue

            performance_list.append({
                'Timestamp': pd.to_datetime(run_value.endtime, unit='ms'),
                'single_best_optimization_score': run_value.cost,
                'single_best_train_score': run_value.additional_info['train_loss'],
            })
        return pd.DataFrame(performance_list)

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        task_name = 'CreateConfigSpace'

        self._stopwatch.start_task(task_name)
        configspace_path = os.path.join(self._backend.temporary_directory, 'space.json')
        configuration_space = util.get_configuration_space(
            dataset_properties=self._datamanager.dataset_properties,
            include=self._include,
            exclude=self._exclude,
            random_state=self._random_state
        )
        # noinspection PyTypeChecker
        self._backend.write_txt_file(configspace_path, cs_json.write(configuration_space), 'Configuration space')
        self._stopwatch.stop_task(task_name)

        return configuration_space

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {}

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a client!
        self._dask_client = None
        self.logging_server = None
        self.stop_logging_server = None
        return self.__dict__
