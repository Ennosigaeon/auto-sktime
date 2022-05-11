import glob
import logging.handlers
import math
import multiprocessing
import numbers
import os
import pickle
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Tuple

import dask.distributed
import numpy as np
import pandas as pd
import pynisher
from distributed import Future
from pandas.core.util.hashing import hash_pandas_object
from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae.base import StatusType
from smac.tae.dask_runner import DaskParallelRunner

from autosktime.automl_common.common.utils.backend import Backend
from autosktime.data import AbstractDataManager
from autosktime.ensembles.selection import EnsembleSelection
from autosktime.ensembles.util import get_ensemble_train, PrefittedEnsembleForecaster
from autosktime.metrics import calculate_loss, BaseMetric
from autosktime.util.dask_single_thread_client import SingleThreadedClient
from sktime.forecasting.compose import EnsembleForecaster

Y_ENSEMBLE = 0
Y_VALID = 1
Y_TEST = 2

MODEL_FN_RE = r'_([0-9]*)_([0-9]*)_([0-9]{1,3}\.[0-9]*)\.npy'


class EnsembleBuilderManager(IncorporateRunResultCallback):
    def __init__(
            self,
            start_time: float,
            time_left_for_ensembles: float,
            backend: Backend,
            dataset_name: str,
            task: int,
            metric: BaseMetric,
            ensemble_size: int,
            ensemble_nbest: Union[int, float],
            max_models_on_disc: Union[int, float] = None,
            seed: int = 1,
            random_state: Union[int, np.random.RandomState] = None,
    ):
        """ SMAC callback to handle ensemble building

        Parameters
        ----------
        start_time: int
            the time when this job was started, to account for any latency in job allocation
        time_left_for_ensembles: int
            How much time is left for the task. Job should finish within this allocated time
        backend: util.backend.Backend
            backend to write and read files
        dataset_name: str
            name of dataset
        task: int
            type of ML task
        metric: str
            name of metric to compute the loss of the given predictions
        ensemble_size: int
            maximal size of ensemble (passed to autosklearn.ensemble.ensemble_selection)
        ensemble_nbest: int/float
            if int: consider only the n best prediction
            if float: consider only this fraction of the best models
            Both wrt to validation predictions
            If performance_range_threshold > 0, might return fewer models
        max_models_on_disc: int
            Defines the maximum number of models that are kept in the disc. As a consequence, it also defines an upper
            bound on the models that can be used in the ensemble.
            If int, it must be greater or equal than 1, and dictates the max number of models to keep.
            If float, it will be interpreted as the max megabytes allowed of disc space. That is, if the number of
            ensemble candidates require more disc space than this float value, the worst models will be deleted to keep
            within this budget. Models and predictions of the worst-performing models will be deleted then.
            If None, the feature is disabled.
        seed: int
            random seed
    Returns
    -------
        List[Tuple[int, float, float, float]]:
            A list with the performance history of this ensemble, of the form
            [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
        """
        self.start_time = start_time
        self.time_left_for_ensembles = time_left_for_ensembles
        self.backend = backend
        self.dataset_name = dataset_name
        self.task = task
        self.metric = metric
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.max_models_on_disc = max_models_on_disc
        self.seed = seed
        self.random_state = random_state

        # Store something similar to SMAC's runhistory
        self.history = []

        # We only submit new ensembles when there is not an active ensemble job
        self.pending_future: Optional[Future] = None

        # The last criteria is the number of iterations
        self.iteration = 0

        # Keep track of when we started to know when we need to finish!
        self.start_time = time.time()

    def __call__(
            self,
            smbo: SMBO,
            run_info: RunInfo,
            result: RunValue,
            time_left: float,
    ):
        if result.status in (StatusType.STOP, StatusType.ABORT) or smbo._stop:
            return

        if isinstance(smbo.tae_runner, DaskParallelRunner):
            self.build_ensemble(smbo.tae_runner.client)
        else:
            self.build_ensemble(SingleThreadedClient())

    def build_ensemble(
            self,
            dask_client: dask.distributed.Client,
    ) -> None:
        logger = logging.getLogger('EnsembleBuilder')

        # Only submit new jobs if the previous ensemble job finished
        if self.pending_future is not None:
            if self.pending_future.done():
                result = self.pending_future.result()
                self.pending_future = None
                if result:
                    logger.debug(f'iteration={self.iteration} @ elapsed_time={time.time() - self.start_time}')
        else:
            # Add the result of the run. On the next while iteration, no references to ensemble builder object, so it
            # should be garbage collected to save memory while waiting for resources. Also, notice how ensemble nbest is
            # returned, so we don't waste iterations testing if the deterministic predictions size can be fitted in
            # memory
            try:
                logger.info(
                    # Log the client to make sure we remain connected to the scheduler
                    f'{self.pending_future}/{dask_client} Started Ensemble builder job at '
                    f'{time.strftime("%Y-%m-%d %H:%M:%S")} for iteration {self.iteration}.'
                )

                self.pending_future = dask_client.submit(
                    _fit_and_return_ensemble,
                    backend=self.backend,
                    dataset_name=self.dataset_name,
                    task_type=self.task,
                    metric=self.metric,
                    ensemble_size=self.ensemble_size,
                    ensemble_nbest=self.ensemble_nbest,
                    max_models_on_disc=self.max_models_on_disc,
                    seed=self.seed,
                    random_state=self.random_state,
                    end_at=self.start_time + self.time_left_for_ensembles,
                    iteration=self.iteration,
                )

                self.iteration += 1
            except Exception as e:
                exception_traceback = traceback.format_exc()
                error_message = repr(e)
                logger.critical(exception_traceback)
                logger.critical(error_message)

    def get_ensemble(self, datamanager: AbstractDataManager) -> Optional[EnsembleForecaster]:
        ensemble: Optional[EnsembleSelection] = self.backend.load_ensemble(self.seed)
        if ensemble is None:
            return None

        forecasters = self.backend.load_models_by_identifiers(ensemble.identifiers_)
        ens = PrefittedEnsembleForecaster(
            forecasters=forecasters.values(),
            weights=ensemble.weights_
        )
        y_train, X_train = get_ensemble_train(datamanager, 0.2)
        ens.fit(y_train, X=X_train)

        return ens


def _fit_and_return_ensemble(
        backend: Backend,
        dataset_name: str,
        task_type: int,
        metric: BaseMetric,
        ensemble_size: int,
        ensemble_nbest: int,
        max_models_on_disc: Union[int, float],
        seed: int,
        end_at: float,
        iteration: int,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> None:
    EnsembleBuilder(
        backend=backend,
        dataset_name=dataset_name,
        task_type=task_type,
        metric=metric,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensemble_nbest,
        max_models_on_disc=max_models_on_disc,
        seed=seed,
        random_state=random_state
    ).run(
        end_at=end_at,
        iteration=iteration,
    )


@dataclass
class ModelLoss:
    seed: int
    num_run: int
    budget: float

    # Lazy keys so far:
    # 0 - not loaded
    # 1 - loaded and in memory
    # 2 - loaded but dropped again
    # 3 - deleted from disk due to space constraints
    loaded: int

    disc_space_cost_mb: float = None
    ens_loss: float = np.inf
    mtime_ens: int = 0


class EnsembleBuilder:
    def __init__(
            self,
            backend: Backend,
            dataset_name: str,
            task_type: int,
            metric: BaseMetric,
            ensemble_size: int = 10,
            ensemble_nbest: Union[int, float] = 100,
            max_models_on_disc: Union[int, float] = None,
            seed: int = 1,
            random_state: Optional[Union[int, np.random.RandomState]] = None,
            use_pynisher: bool = False
    ):
        self.backend = backend
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.metric = metric
        self.ensemble_size = ensemble_size

        if isinstance(ensemble_nbest, numbers.Integral) and ensemble_nbest < 1:
            raise ValueError(f'Integer ensemble_nbest has to be larger 1: {ensemble_nbest}')
        elif not isinstance(ensemble_nbest, numbers.Integral):
            if not 0 < ensemble_nbest <= 1:
                raise ValueError(f'Float ensemble_nbest best has to be > 0 and <= 1: {ensemble_nbest}')

        self.ensemble_nbest = ensemble_nbest

        # max_models_on_disc can be a float, in such case we need to remember the user specified Megabytes and translate
        # this to max number of ensemble models. max_resident_models keeps the maximum number of models on disc
        if max_models_on_disc is not None and max_models_on_disc < 0:
            raise ValueError('max_models_on_disc has to be a positive number or None')
        self.max_models_on_disc = max_models_on_disc
        self.max_resident_models = None

        self.seed = seed
        self.random_state = random_state
        self.use_pynisher = use_pynisher

        self.logger = logging.getLogger('EnsembleBuilder')

        self.model_fn_re = re.compile(MODEL_FN_RE)

        self.last_hash = None  # hash of ensemble training data
        self.y_true_ensemble = None

        self.read_losses: Dict[str, ModelLoss] = {}
        self.predictions: Dict[str, Optional[pd.Series]] = {}

        # Depending on the dataset dimensions, regenerating every iteration, the predictions losses for self.read_preds
        # is too computationally expensive as the ensemble builder is stateless (every time the ensemble builder gets
        # resources from dask, it builds this object from scratch) we save the state of this dictionary to memory
        # and read it if available
        self.ensemble_memory_file = os.path.join(self.backend.internals_directory, 'ensemble_read_preds.pkl')
        if os.path.exists(self.ensemble_memory_file):
            try:
                with (open(self.ensemble_memory_file, 'rb')) as memory:
                    predictions, self.last_hash = pickle.load(memory)
                    self.predictions: Dict[str, Optional[pd.Series]] = predictions
            except Exception as e:
                self.logger.warning(
                    'Could not load the previous iterations of ensemble_builder predictions.'
                    f'This might impact the quality of the run. Exception={e} {traceback.format_exc()}'
                )
        self.ensemble_loss_file = os.path.join(self.backend.internals_directory, 'ensemble_read_losses.pkl')
        if os.path.exists(self.ensemble_loss_file):
            try:
                with (open(self.ensemble_loss_file, 'rb')) as memory:
                    self.read_losses: Dict[str, ModelLoss] = pickle.load(memory)
            except Exception as e:
                self.logger.warning(
                    'Could not load the previous iterations of ensemble_builder losses.'
                    f'This might impact the quality of the run. Exception={e} {traceback.format_exc()}'
                )

        self.validation_performance_ = np.inf

    def run(
            self,
            iteration: int,
            time_left: Optional[float] = None,
            end_at: Optional[float] = None,
            time_buffer: int = 5,
    ) -> None:
        if time_left is None and end_at is None:
            raise ValueError('Must provide either time_left or end_at.')
        elif time_left is not None and end_at is not None:
            raise ValueError('Cannot provide both time_left and end_at.')

        if self.use_pynisher:
            if time_left is not None:
                time_elapsed = time.time() - time.time()
                time_left -= time_elapsed
            else:
                current_time = time.time()
                time_left = end_at - current_time
            wall_time_in_s = int(time_left - time_buffer)

            context = multiprocessing.get_context()
            safe_ensemble_script = pynisher.enforce_limits(
                wall_time_in_s=wall_time_in_s,
                logger=self.logger,
                context=context,
            )(self._run)
            safe_ensemble_script(time_left, iteration)
        else:
            try:
                self._run(np.inf, iteration)
            except Exception as e:
                self.logger.exception(e)

    def _run(self, time_left: float, iteration: int) -> None:
        # populates self.read_preds and self.read_losses
        if not self.compute_loss_per_model():
            return

        # Only the models with the n_best predictions are candidates to be in the ensemble
        candidate_models = self.get_n_best_preds()
        if not candidate_models:  # no candidates yet
            return

        # train ensemble
        ensemble = self.fit_ensemble(selected_keys=candidate_models)

        # Save the ensemble for later use in the main auto-sktime module!
        if ensemble is not None:
            self.backend.save_ensemble(ensemble, iteration, self.seed)

        # Delete files of non-candidate models - can only be done after fitting the ensemble and saving it to disc, so
        # we do not accidentally delete models in the previous ensemble
        if self.max_resident_models is not None:
            self._delete_excess_models(selected_keys=candidate_models)

        # Save the read losses status for the next iteration
        with open(self.ensemble_loss_file, 'wb') as memory:
            pickle.dump(self.read_losses, memory)

        # The loaded predictions and the hash can only be saved after the ensemble has been built, because the hash is
        # computed during the construction of the ensemble
        with open(self.ensemble_memory_file, 'wb') as memory:
            pickle.dump((self.predictions, self.last_hash), memory)

    def compute_loss_per_model(self):
        self.logger.debug('Read ensemble data set predictions')

        if self.y_true_ensemble is None:
            try:
                self.y_true_ensemble = self.backend.load_targets_ensemble()
            except FileNotFoundError:
                self.logger.debug(f'Could not find true targets on ensemble data set: {traceback.format_exc()}')
                return False

        pred_path = os.path.join(
            glob.escape(self.backend.get_runs_directory()),
            f'{self.seed}_*_*',
            f'predictions_ensemble_{self.seed}_*_*.npy*'
        )

        y_ens_files = [y_ens_file for y_ens_file in glob.glob(pred_path) if y_ens_file.endswith('.npy')]
        self.y_ens_files = y_ens_files

        # no predictions so far -- no files
        if len(self.y_ens_files) == 0:
            self.logger.debug(f'Found no prediction files on ensemble data set: {pred_path}')
            return False

        # First sort files chronologically
        to_read = []
        for y_ens_fn in self.y_ens_files:
            match = self.model_fn_re.search(y_ens_fn)
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))
            mtime = os.path.getmtime(y_ens_fn)

            to_read.append([y_ens_fn, match, _seed, _num_run, _budget, mtime])

        n_read_files = 0
        # Now read file wrt to num_run
        for y_ens_fn, match, _seed, _num_run, _budget, mtime in sorted(to_read, key=lambda x: x[5]):
            if not y_ens_fn.endswith('.npy'):
                self.logger.info(f'Error loading file (not .npy): {y_ens_fn}')
                continue

            if not self.read_losses.get(y_ens_fn):
                self.read_losses[y_ens_fn] = ModelLoss(seed=_seed, num_run=_num_run, budget=_budget, loaded=0)
            if y_ens_fn not in self.predictions:
                self.predictions[y_ens_fn] = None

            if self.read_losses[y_ens_fn].mtime_ens == mtime:
                # same time stamp; nothing changed;
                continue

            # actually read the predictions and compute their respective loss
            try:
                y_ensemble = self._read_np_fn(y_ens_fn)
                loss = calculate_loss(solution=self.y_true_ensemble,
                                      prediction=y_ensemble,
                                      task_type=self.task_type,
                                      metric=self.metric)

                self.read_losses[y_ens_fn].ens_loss = loss

                # It is not needed to create the object here. To save memory, we just compute the loss.
                self.read_losses[y_ens_fn].mtime_ens = os.path.getmtime(y_ens_fn)
                self.read_losses[y_ens_fn].loaded = 2

                if self.max_models_on_disc is not None and not isinstance(self.max_models_on_disc, numbers.Integral):
                    self.read_losses[y_ens_fn].disc_space_cost_mb = self._get_disk_consumption(y_ens_fn)

                n_read_files += 1

            except Exception:
                self.logger.warning(f'Error loading {y_ens_fn}: {traceback.format_exc()}')
                self.read_losses[y_ens_fn].ens_loss = np.inf

        self.logger.debug(
            f'Done reading {n_read_files} new prediction files. '
            f'Loaded {np.sum([pred.loaded > 0 for pred in self.read_losses.values()])} predictions in total.')
        return True

    def _get_disk_consumption(self, pred_path: str):
        match = self.model_fn_re.search(pred_path)
        if not match:
            raise ValueError(f'Invalid path format {pred_path}')
        _seed = int(match.group(1))
        _num_run = int(match.group(2))
        _budget = float(match.group(3))

        stored_files_for_run = os.listdir(self.backend.get_numrun_directory(_seed, _num_run, _budget))
        stored_files_for_run = [
            os.path.join(self.backend.get_numrun_directory(_seed, _num_run, _budget), file_name)
            for file_name in stored_files_for_run]
        this_model_cost = sum([os.path.getsize(path) for path in stored_files_for_run])

        # get the megabytes
        return round(this_model_cost / math.pow(1024, 2), 2)

    def get_n_best_preds(self) -> List[str]:
        """
        get best n predictions (i.e., keys of self.read_losses)
        according to the loss on the 'ensemble set'
        n: self.ensemble_nbest

        Side effects:
            -> Define the n-best models to use in ensemble
            -> Only the best models are loaded
            -> Any model that is not best is candidate to deletion if max_models_on_disc is exceeded.
        """
        sorted_keys = self._get_list_of_sorted_preds()

        # reload predictions if losses changed over time and a model is considered to be in the top models again!
        if not isinstance(self.ensemble_nbest, numbers.Integral):
            keep_nbest = max(1, min(len(sorted_keys), int(len(sorted_keys) * self.ensemble_nbest)))
            self.logger.debug(
                f'Library pruning: using only top {self.ensemble_nbest * 100:.2f} percent of the models for ensemble '
                f'({keep_nbest}/{len(sorted_keys)})')
        else:
            keep_nbest = min(self.ensemble_nbest, len(sorted_keys))
            self.logger.debug(f'Library Pruning: using for ensemble {keep_nbest}/{len(sorted_keys)} models')

        # One can only read at most max_models_on_disc models
        if isinstance(self.max_models_on_disc, numbers.Integral):
            self.max_resident_models = self.max_models_on_disc
        elif isinstance(self.max_models_on_disc, numbers.Real):
            consumption = [[v.ens_loss, v.disc_space_cost_mb] for v in self.read_losses.values()
                           if v.disc_space_cost_mb is not None]
            largest_consumption = max(c[1] for c in consumption)

            # We are pessimistic with the consumption limit indicated by max_models_on_disc by 1 model. Such model
            # is assumed to spend largest_consumption megabytes
            if (sum(c[1] for c in consumption) + largest_consumption) > self.max_models_on_disc:
                # just leave the best -- smaller is better!
                # This list is in descending order, to preserve the best models
                sorted_cum_consumption = np.cumsum([c[1] for c in list(sorted(consumption))]) + largest_consumption
                max_models = np.argmax(sorted_cum_consumption > self.max_models_on_disc)

                # Make sure that at least 1 model survives
                self.max_resident_models = max(1, int(max_models))
                self.logger.warning(
                    f'Limiting num of models as accumulated={(sum(c[1] for c in consumption) + largest_consumption)} '
                    f'worst={largest_consumption} num_models={self.max_resident_models}')
        else:
            self.max_resident_models = None

        if self.max_resident_models is not None and keep_nbest > self.max_resident_models:
            self.logger.debug(
                f'Restricting number of models to {self.max_resident_models}/{keep_nbest} due to max_models_on_disc')
            keep_nbest = self.max_resident_models

        # reduce to keys
        sorted_keys = list(map(lambda x: x[0], sorted_keys))

        # remove loaded predictions for non-winning models
        for k in sorted_keys[keep_nbest:]:
            if k in self.predictions:
                self.predictions[k] = None
            if self.read_losses[k].loaded == 1:
                self.logger.debug(
                    f'Dropping model {k} ({self.read_losses[k].seed},{self.read_losses[k].num_run}) with loss '
                    f'{self.read_losses[k].ens_loss}.')
                self.read_losses[k].loaded = 2

        # Load the predictions for the winning
        for k in sorted_keys[:keep_nbest]:
            if (k not in self.predictions or self.predictions[k] is None) and self.read_losses[k].loaded != 3:
                self.predictions[k] = self._read_np_fn(k)
                self.read_losses[k].loaded = 1

        # return keys of self.read_losses with lowest losses
        return sorted_keys[:keep_nbest]

    def _get_list_of_sorted_preds(self) -> List[Tuple[str, float, int]]:
        # Sort by loss - smaller is better!
        sorted_keys = list(sorted(
            [(k, v.ens_loss, v.num_run) for k, v in self.read_losses.items()],
            # Sort by loss as priority 1 and then by num_run on an ascending order. We want small num_run first
            key=lambda x: (x[1], x[2]),
        ))
        return sorted_keys

    def fit_ensemble(self, selected_keys: List[str]) -> Optional[EnsembleSelection]:
        predictions_train = [self.predictions[k] for k in selected_keys]
        include_num_runs = [(self.read_losses[k].seed, self.read_losses[k].num_run, self.read_losses[k].budget)
                            for k in selected_keys]

        # check hash if ensemble training data changed
        current_hash = ''.join([f'{hash_pandas_object(pred).sum()}' for pred in predictions_train])

        if self.last_hash == current_hash:
            self.logger.debug(
                f'No new model predictions selected -- skip ensemble building -- current performance: '
                f'{self.validation_performance_}')
            return None
        self.last_hash = current_hash

        ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            task_type=self.task_type,
            metric=self.metric,
            random_state=self.random_state,
        )

        try:
            self.logger.debug(f'Fitting the ensemble on {len(predictions_train)} models.')
            start_time = time.time()
            ensemble.fit(predictions_train, self.y_true_ensemble, include_num_runs)
            end_time = time.time()
            self.logger.debug(f'Fitting the ensemble took {end_time - start_time:.2f} seconds.')
            self.logger.info(ensemble)
            self.validation_performance_ = min(self.validation_performance_, ensemble.get_validation_performance())
        except ValueError:
            self.logger.error(f'Caught ValueError: {traceback.format_exc()}')
            return None
        except IndexError:
            self.logger.error(f'Caught IndexError: {traceback.format_exc()}')
            return None
        finally:
            # Explicitly free memory
            del predictions_train

        return ensemble

    def _delete_excess_models(self, selected_keys: List[str]):
        """
        Deletes models excess models on disc. self.max_models_on_disc defines the upper limit on how many models to
        keep. Any additional model with a worst loss than the top self.max_models_on_disc is deleted.
        """

        worst_models = [(v.ens_loss, k) for k, v in self.read_losses.items() if v.loaded != 3]
        worst_models = list(sorted(worst_models))[self.max_resident_models:]

        for _, pred_path in worst_models:
            if pred_path in selected_keys:
                # Safety-net to prevent deleting used models, should not be necessary
                continue

            match = self.model_fn_re.search(pred_path)
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))

            directory = self.backend.get_numrun_directory(_seed, _num_run, _budget)
            try:
                os.rename(directory, f'{directory}.old')
                shutil.rmtree(f'{directory}.old')
                self.logger.info(f'Deleted files of non-candidate model {pred_path}')
                self.read_losses[pred_path].disc_space_cost_mb = None
                self.read_losses[pred_path].loaded = 3
                self.read_losses[pred_path].ens_loss = np.inf
            except Exception as e:
                self.logger.error(f'Failed to delete non-candidate model {pred_path} due to error {e}')

    @staticmethod
    def _read_np_fn(path) -> pd.Series:
        predictions = np.load(path, allow_pickle=True).astype(dtype=np.float32)
        # noinspection PyTypeChecker
        return predictions
