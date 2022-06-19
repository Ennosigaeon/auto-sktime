import argparse
import glob
import itertools
import os
import pathlib
import tempfile

import pandas as pd
from smac.tae import StatusType

from autosktime.automl import AutoML
from autosktime.constants import UNIVARIATE_FORECAST, TASK_TYPES_TO_STRING
from autosktime.data.benchmark.m4 import load_timeseries
from autosktime.metrics import calculate_loss, STRING_TO_METRIC


def run_auto_sktime():
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
    dataset = args.task_id
    seed = args.seed
    metric = args.metric
    # TODO task is hard-coded
    task = UNIVARIATE_FORECAST

    y_train, y_test = load_timeseries(dataset)

    tempdir = tempfile.mkdtemp()
    autosktime_directory = os.path.join(tempdir, "dir")

    automl = AutoML(
        time_left_for_this_task=time_limit,
        per_run_time_limit=per_run_time_limit,
        metric=STRING_TO_METRIC[metric],
        ensemble_size=0,
        ensemble_nbest=0,
        seed=seed,
        memory_limit=3072,
        hp_priors=False,
        num_metalearning_configs=-1,
        # resampling_strategy='sliding-window',
        # resampling_strategy_arguments={'folds': 2 if is_test else 5},
        delete_tmp_folder_after_terminate=False,
        temporary_directory=autosktime_directory

    )
    automl.fit(y_train, dataset_name=dataset)

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
            'dataset': dataset,
            'id': key.config_id,
            'train_score': float(train_performance),
            'test_score': float(test_performance),
            'values': run_history.ids_config[key.config_id].get_array()
        })
        print(f'Finished validating configuration {i + 1}/{len(run_history.data)}')
    print('Finished to validate configurations')

    print('*' * 80)
    print('Starting to copy results to configuration directory', flush=True)
    configuration_output_dir = os.path.join(working_directory, 'meta', 'raw', f'{TASK_TYPES_TO_STRING[task]}_{metric}')
    os.makedirs(configuration_output_dir, exist_ok=True)
    pd.DataFrame(validated_configurations) \
        .sort_values(by=['test_score']) \
        .to_csv(os.path.join(configuration_output_dir, f'{dataset}.csv'), index=False)
    pd.concat([y_train, y_test], axis=0) \
        .reset_index(drop=True) \
        .to_pickle(os.path.join(configuration_output_dir, f'{dataset}.npy.gz'))
    print('*' * 80)
    print('Finished copying the configuration directory')


def aggregate_results():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-directory", type=str, required=True)

    args, _ = parser.parse_known_args()
    working_directory = os.path.join(args.working_directory, 'meta')

    metadata_sets = itertools.product([UNIVARIATE_FORECAST], STRING_TO_METRIC.keys())
    for task, metric in metadata_sets:
        print(TASK_TYPES_TO_STRING[task], metric)
        input_directory = os.path.join(working_directory, 'raw', f'{TASK_TYPES_TO_STRING[task]}_{metric}')

        config_files = sorted(glob.glob(os.path.join(input_directory, '*.csv')))
        configs = []
        timeseries = {}
        for config_file in config_files:
            configs.append(pd.read_csv(config_file))
            timeseries[pathlib.Path(config_file).stem] = pd.read_pickle(config_file.replace('.csv', '.npy.gz'))

        if len(timeseries) == 0:
            print(f'No data found for {TASK_TYPES_TO_STRING[task]} {metric}')
            continue

        configs = pd.concat(configs)
        timeseries = pd.DataFrame(timeseries)

        output_dir = os.path.join(working_directory, 'agg', f'{TASK_TYPES_TO_STRING[task]}_{metric}')
        os.makedirs(output_dir, exist_ok=True)
        configs.to_csv(os.path.join(output_dir, 'configurations.csv'), index=False)
        timeseries.to_pickle(os.path.join(output_dir, 'timeseries.npy.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, required=True, choices=['run', 'aggregate'])
    args, _ = parser.parse_known_args()

    if args.step == 'run':
        run_auto_sktime()
    elif args.step == 'aggregate':
        aggregate_results()
