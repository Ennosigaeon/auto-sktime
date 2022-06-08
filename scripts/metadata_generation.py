import argparse
import os
import tempfile

import pandas as pd
from smac.tae import StatusType

from autosktime.automl import AutoML
from autosktime.data.benchmark.m4 import load_timeseries
from autosktime.metrics import calculate_loss, all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working-directory', type=str, required=True)
    parser.add_argument('--time-limit', type=int, required=True)
    parser.add_argument('--per-run-time-limit', type=int, required=True)
    parser.add_argument('--task-id', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('-s', '--seed', type=int, required=True)
    args, _ = parser.parse_known_args()

    working_directory = args.working_directory
    time_limit = args.time_limit
    per_run_time_limit = args.per_run_time_limit
    task_id = args.task_id
    seed = args.seed
    metric = args.metric
    # TODO task is hard-coded
    task = 'univariate'

    y_train, y_test = load_timeseries(task_id)

    tempdir = tempfile.mkdtemp()
    autosktime_directory = os.path.join(tempdir, "dir")

    automl = AutoML(
        time_left_for_this_task=time_limit,
        per_run_time_limit=per_run_time_limit,
        metric=all_metrics[metric],
        ensemble_size=0,
        ensemble_nbest=0,
        seed=seed,
        memory_limit=3072,
        # resampling_strategy='sliding-window',
        # resampling_strategy_arguments={'folds': 2 if is_test else 5},
        delete_tmp_folder_after_terminate=False,
        temporary_directory=autosktime_directory

    )
    automl.fit(y_train, dataset_name=task_id)

    print('Starting to validate configurations')
    if automl._resampling_strategy in ['sliding-window']:
        load_function = automl._backend.load_cv_model_by_seed_and_id_and_budget
    else:
        load_function = automl._backend.load_model_by_seed_and_id_and_budget

    run_history = automl.runhistory_
    validated_configurations = []
    for i, (key, value) in enumerate(run_history.data.items()):
        print(f'Starting to validate configuration {i + 1}/{len(run_history.data)}')

        if value.status == StatusType.SUCCESS:
            pipeline = load_function(
                seed=automl._seed,
                idx=key.config_id,
                budget=key.budget,
            )
            y_pred = pipeline.predict(y_test.index)
            test_performance = calculate_loss(y_test, y_pred, automl._task, automl._metric)
            train_performance = value.cost
        else:
            test_performance = value.cost
            train_performance = value.cost
        validated_configurations.append({
            'id': key.config_id,
            'train_score': float(train_performance),
            'test_score': float(test_performance),
            **run_history.ids_config[key.config_id].get_dictionary()
        })
        print(f'Finished validating configuration {i + 1}/{len(run_history.data)}')
    print('Finished to validate configurations')

    print('*' * 80)
    print('Starting to copy results to configuration directory', flush=True)
    configuration_output_dir = os.path.join(working_directory, 'configuration', f'{task}_{metric}')
    os.makedirs(configuration_output_dir, exist_ok=True)
    pd.DataFrame(validated_configurations).to_csv(os.path.join(configuration_output_dir, f'{task_id}.csv'), index=False)
    pd.concat([y_train, y_test], axis=1).to_pickle(os.path.join(configuration_output_dir, f'{task_id}.npy.gz'))
    print('*' * 80)
    print('Finished copying the configuration directory')


if __name__ == '__main__':
    main()
