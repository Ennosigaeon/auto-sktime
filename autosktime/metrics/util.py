import os

from smac.runhistory.runhistory import RunHistory

from autosktime.automl_common.common.utils.backend import Backend
from autosktime.metrics import _BoundedMetricMixin
from sktime.forecasting.base import BaseForecaster


def get_incumbent(
        run_history: RunHistory,
        metric: _BoundedMetricMixin,
        backend: Backend,
        seed: int
) -> BaseForecaster:
    """
    This method parses the run history, to identify the best performing model.
    """
    best_model_identifier = None
    best_model_score = metric.worst_possible_result

    for run_key in run_history.data.keys():
        run_value = run_history.data[run_key]
        score = metric.optimum - run_value.cost

        if score < best_model_score:
            # Make sure that the individual best model actually exists
            model_dir = backend.get_numrun_directory(seed, run_key.config_id, run_key.budget)
            model_file_name = backend.get_model_filename(seed, run_key.config_id, run_key.budget)
            file_path = os.path.join(model_dir, model_file_name)
            if not os.path.exists(file_path):
                continue

            best_model_identifier = (seed, run_value.additional_info['num_run'], run_key.budget)
            best_model_score = score

    if not best_model_identifier:
        raise ValueError(
            "No valid model found in run history. This means smac was not able to fit a valid model. Please check "
            "the log file for errors."
        )

    # noinspection PyTypeChecker
    return backend.load_model_by_seed_and_id_and_budget(*best_model_identifier)
