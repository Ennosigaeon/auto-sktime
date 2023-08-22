import inspect
from typing import Optional

import pandas as pd
from autots import AutoTS


# This code is not debuggable due to
# https://stackoverflow.com/questions/70929565/in-debug-using-pandas-before-importing-from-scipy-generates-type-error-on-impor

def evaluate_autots(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
        random: bool = False
):
    if isinstance(y, pd.Series):
        y = y.to_frame()
    df = y.reset_index()

    if X_train is not None:
        for col in X_train:
            df[col] = X_train[col].values

    res = []
    res_ints = []

    for col in y.columns:
        model = CustomAutoTS(
            forecast_length=fh,
            prediction_interval=0.5,
            max_generations=500,
            num_validations=0,
            generation_timeout=max(1, max_duration // 60 // len(y.columns)),
            random_seed=seed,
            initial_template='Random' if random else 'General+Random'
        )
        model = model.fit(
            df,
            date_col='index',
            value_col=col,
            id_col='series' if isinstance(y.index, pd.MultiIndex) else None
        )
        output = model.predict()

        if isinstance(y.index, pd.MultiIndex):
            predictions = pd.melt(output.forecast.reset_index(), id_vars='index', value_vars=output.forecast.columns).set_index(['variable', 'index'])
            lower_forecast = pd.melt(output.lower_forecast.reset_index(), id_vars='index',
                                     value_vars=output.lower_forecast.columns).set_index(['variable', 'index'])
            upper_forecast = pd.melt(output.upper_forecast.reset_index(), id_vars='index',
                                     value_vars=output.upper_forecast.columns).set_index(['variable', 'index'])
        else:
            predictions = output.forecast
            lower_forecast = output.lower_forecast
            upper_forecast = output.upper_forecast

        predictions = predictions.rename(columns={'value': col})

        y_pred_ints = pd.DataFrame(
            upper_forecast.join(lower_forecast, rsuffix='r').values,
            columns=pd.MultiIndex.from_tuples([(f'Coverage_{col}', 0.5, 'lower'), (f'Coverage_{col}', 0.5, 'upper')]),
            index=predictions.index
        )

        res.append(predictions)
        res_ints.append(y_pred_ints)

    return pd.concat(res, axis=1), pd.concat(res_ints, axis=1)


def evaluate_autots_random(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
        random: bool = True
):
    return evaluate_autots(y, X_train, X_test, fh, max_duration, name, seed, random)


import datetime

import numpy as np
import pandas as pd
import copy
import json
import sys
import traceback as tb

from autots.tools.shaping import subset_series, simple_train_test_split
from autots.evaluator.auto_model import (
    TemplateEvalObject,
    TemplateWizard,
    generate_score,
    model_forecast,
    validation_aggregation, _ps_metric, create_model_id,
)


def TemplateWizard(
        template,
        df_train,
        df_test,
        weights,
        model_count: int = 0,
        ensemble: list = ["mosaic", "distance"],
        forecast_length: int = 14,
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        no_negatives: bool = False,
        constraint: float = None,
        future_regressor_train=None,
        future_regressor_forecast=None,
        holiday_country: str = 'US',
        startTimeStamps=None,
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = None,
        validation_round: int = 0,
        current_generation: int = 0,
        max_generations: str = "0",
        model_interrupt: bool = False,
        grouping_ids=None,
        template_cols: list = [
            'Model',
            'ModelParameters',
            'TransformationParameters',
            'Ensemble',
        ],
        traceback: bool = False,
        current_model_file: str = None,
        mosaic_list=[
            'mosaic-window',
            'mosaic',
            'mosaic_crosshair',
            "mosaic_window",
            "mosaic-crosshair",
        ],

        # auto-sktime
        start_time=None,
        generation_timeout=None,
):
    """
    Take Template, returns Results.

    There are some who call me... Tim. - Python

    Args:
        template (pandas.DataFrame): containing model str, and json of transformations and hyperparamters
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
        df_test (pandas.DataFrame): dataframe of actual values of (forecast length * n series)
        weights (dict): key = column/series_id, value = weight
        ensemble (list): list of ensemble types to prepare metric collection
        forecast_length (int): number of periods to forecast
        transformation_dict (dict): a dictionary of outlier, fillNA, and transformation methods to be used
        model_str (str): a string to be direct to the appropriate model, used in ModelMonster
        frequency (str): str representing frequency alias of time series
        prediction_interval (float): width of errors (note: rarely do the intervals accurately match the % asked for...)
        no_negatives (bool): whether to force all forecasts to be > 0
        constraint (float): when not None, use this value * data st dev above max or below min for constraining forecast values.
        future_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        future_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.
        startTimeStamps (pd.Series): index (series_ids), columns (Datetime of First start of series)
        validation_round (int): int passed to record current validation.
        current_generation (int): info to pass to print statements
        max_generations (str): info to pass to print statements
        model_interrupt (bool): if True, keyboard interrupts are caught and only break current model eval.
        template_cols (list): column names of columns used as model template
        traceback (bool): include tracebook over just error representation
        current_model_file (str): file path to write to disk of current model params (for debugging if computer crashes). .json is appended

    Returns:
        TemplateEvalObject
    """
    best_smape = float("inf")
    template_result = TemplateEvalObject(
        per_series_mae=[],
        per_series_made=[],
        per_series_contour=[],
        per_series_rmse=[],
        per_series_spl=[],
        per_series_mle=[],
        per_series_imle=[],
        per_series_maxe=[],
        per_series_oda=[],
        per_series_mqae=[],
        per_series_dwae=[],
        per_series_ewmae=[],
        per_series_uwmse=[],
        per_series_smoothness=[],
    )
    template_result.model_count = model_count
    if isinstance(template, pd.Series):
        template = template.to_frame()
    if verbose > 1:
        try:
            from psutil import virtual_memory
        except Exception:

            class MemObjecty(object):
                def __init__(self):
                    self.percent = np.nan

            def virtual_memory():
                return MemObjecty()

    # template = unpack_ensemble_models(template, template_cols, keep_ensemble = False)

    # precompute scaler to save a few miliseconds (saves very little time)
    scaler = np.nanmean(np.abs(np.diff(df_train[-100:], axis=0)), axis=0)
    fill_val = np.nanmax(scaler)
    fill_val = fill_val if fill_val > 0 else 1
    scaler[scaler == 0] = fill_val
    scaler[np.isnan(scaler)] = fill_val

    template_dict = template.to_dict('records')
    # minor speedup with one less copy per eval by assuring arrays at this level
    actuals = np.asarray(df_test)
    df_trn_arr = np.asarray(df_train)

    # auto-sktime
    print(
        "###########################",
        (pd.Timestamp.now() - start_time).total_seconds() / 60 if start_time is not None else None
    )

    for row in template_dict:
        # auto-sktime
        if start_time is not None and generation_timeout is not None:
            passedTime = (pd.Timestamp.now() - start_time).total_seconds() / 60
            if passedTime > generation_timeout:
                print('Aborting fitting due to timeout')
                break

        template_start_time = datetime.datetime.now()
        try:
            model_str = row['Model']
            parameter_dict = json.loads(row['ModelParameters'])
            transformation_dict = json.loads(row['TransformationParameters'])
            ensemble_input = row['Ensemble']
            template_result.model_count += 1
            if verbose > 0:
                if validation_round >= 1:
                    base_print = (
                        "Model Number: {} of {} with model {} for Validation {}".format(
                            str(template_result.model_count),
                            template.shape[0],
                            model_str,
                            str(validation_round),
                        )
                    )
                else:
                    base_print = (
                        "Model Number: {} with model {} in generation {} of {}".format(
                            str(template_result.model_count),
                            model_str,
                            str(current_generation),
                            str(max_generations),
                        )
                    )
                if verbose > 1:
                    print(
                        base_print
                        + " with params {} and transformations {}".format(
                            json.dumps(parameter_dict),
                            json.dumps(transformation_dict),
                        )
                    )
                else:
                    print(base_print)
            df_forecast = model_forecast(
                model_name=row['Model'],
                model_param_dict=row['ModelParameters'],
                model_transform_dict=row['TransformationParameters'],
                df_train=df_train,
                forecast_length=forecast_length,
                frequency=frequency,
                prediction_interval=prediction_interval,
                no_negatives=no_negatives,
                constraint=constraint,
                future_regressor_train=future_regressor_train,
                future_regressor_forecast=future_regressor_forecast,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
                template_cols=template_cols,
                current_model_file=current_model_file,
                model_count=template_result.model_count,
            )
            if verbose > 1:
                post_memory_percent = virtual_memory().percent

            per_ts = True if 'distance' in ensemble else False
            model_error = df_forecast.evaluate(
                actuals,
                series_weights=weights,
                df_train=df_trn_arr,
                per_timestamp_errors=per_ts,
                scaler=scaler,
            )
            if validation_round >= 1 and verbose > 0:
                round_smape = model_error.avg_metrics['smape'].round(2)
                validation_accuracy_print = "{} - {} with avg smape {}: ".format(
                    str(template_result.model_count),
                    model_str,
                    round_smape,
                )
                if round_smape < best_smape:
                    best_smape = round_smape
                    try:
                        print("\U0001F4C8 " + validation_accuracy_print)
                    except Exception:
                        print(validation_accuracy_print)
                else:
                    print(validation_accuracy_print)
            model_id = create_model_id(
                df_forecast.model_name,
                df_forecast.model_parameters,
                df_forecast.transformation_parameters,
            )
            result = pd.DataFrame(
                {
                    'ID': model_id,
                    'Model': df_forecast.model_name,
                    'ModelParameters': json.dumps(df_forecast.model_parameters),
                    'TransformationParameters': json.dumps(
                        df_forecast.transformation_parameters
                    ),
                    'TransformationRuntime': df_forecast.transformation_runtime,
                    'FitRuntime': df_forecast.fit_runtime,
                    'PredictRuntime': df_forecast.predict_runtime,
                    'TotalRuntime': datetime.datetime.now() - template_start_time,
                    'Ensemble': ensemble_input,
                    'Exceptions': np.nan,
                    'Runs': 1,
                    'Generation': current_generation,
                    'ValidationRound': validation_round,
                    'ValidationStartDate': df_forecast.forecast.index[0],
                },
                index=[0],
            )
            if verbose > 1:
                result['PostMemoryPercent'] = post_memory_percent
            a = pd.DataFrame(
                model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')
            ).transpose()
            result = pd.concat(
                [result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis=1
            )
            template_result.model_results = pd.concat(
                [template_result.model_results, result],
                axis=0,
                ignore_index=True,
                sort=False,
            ).reset_index(drop=True)

            ps_metric = model_error.per_series_metrics

            template_result.per_series_mae.append(
                _ps_metric(ps_metric, 'mae', model_id)
            )
            template_result.per_series_made.append(
                _ps_metric(ps_metric, 'made', model_id)
            )
            template_result.per_series_contour.append(
                _ps_metric(ps_metric, 'contour', model_id)
            )
            template_result.per_series_rmse.append(
                _ps_metric(ps_metric, 'rmse', model_id)
            )
            template_result.per_series_spl.append(
                _ps_metric(ps_metric, 'spl', model_id)
            )
            template_result.per_series_mle.append(
                _ps_metric(ps_metric, 'mle', model_id)
            )
            template_result.per_series_imle.append(
                _ps_metric(ps_metric, 'imle', model_id)
            )
            template_result.per_series_maxe.append(
                _ps_metric(ps_metric, 'maxe', model_id)
            )
            template_result.per_series_oda.append(
                _ps_metric(ps_metric, 'oda', model_id)
            )
            template_result.per_series_mqae.append(
                _ps_metric(ps_metric, 'mqae', model_id)
            )
            template_result.per_series_dwae.append(
                _ps_metric(ps_metric, 'dwae', model_id)
            )
            template_result.per_series_ewmae.append(
                _ps_metric(ps_metric, 'ewmae', model_id)
            )
            template_result.per_series_uwmse.append(
                _ps_metric(ps_metric, 'uwmse', model_id)
            )
            template_result.per_series_smoothness.append(
                _ps_metric(ps_metric, 'smoothness', model_id)
            )
            if 'distance' in ensemble:
                cur_smape = model_error.per_timestamp.loc['weighted_smape']
                cur_smape = pd.DataFrame(cur_smape).transpose()
                cur_smape.index = [model_id]
                template_result.per_timestamp_smape = pd.concat(
                    [template_result.per_timestamp_smape, cur_smape], axis=0
                )
            if any([x in mosaic_list for x in ensemble]):
                template_result.full_mae_errors.extend([model_error.full_mae_errors])
                template_result.squared_errors.extend([model_error.squared_errors])
                template_result.full_pl_errors.extend(
                    [model_error.upper_pl + model_error.lower_pl]
                )
                template_result.full_mae_ids.extend([model_id])

        except KeyboardInterrupt:
            if model_interrupt:
                fit_runtime = datetime.datetime.now() - template_start_time
                result = pd.DataFrame(
                    {
                        'ID': create_model_id(
                            model_str, parameter_dict, transformation_dict
                        ),
                        'Model': model_str,
                        'ModelParameters': json.dumps(parameter_dict),
                        'TransformationParameters': json.dumps(transformation_dict),
                        'Ensemble': ensemble_input,
                        'TransformationRuntime': datetime.timedelta(0),
                        'FitRuntime': fit_runtime,
                        'PredictRuntime': datetime.timedelta(0),
                        'TotalRuntime': fit_runtime,
                        'Exceptions': "KeyboardInterrupt by user",
                        'Runs': 1,
                        'Generation': current_generation,
                        'ValidationRound': validation_round,
                    },
                    index=[0],
                )
                template_result.model_results = pd.concat(
                    [template_result.model_results, result],
                    axis=0,
                    ignore_index=True,
                    sort=False,
                ).reset_index(drop=True)
                if model_interrupt == "end_generation" and current_generation > 0:
                    break
            else:
                sys.stdout.flush()
                raise KeyboardInterrupt
        except Exception as e:
            if verbose >= 0:
                if traceback:
                    print(
                        'Template Eval Error: {} in model {} in generation {}: {}'.format(
                            ''.join(tb.format_exception(None, e, e.__traceback__)),
                            template_result.model_count,
                            str(current_generation),
                            model_str,
                        )
                    )
                else:
                    print(
                        'Template Eval Error: {} in model {} in generation {}: {}'.format(
                            (repr(e)),
                            template_result.model_count,
                            str(current_generation),
                            model_str,
                        )
                    )
            fit_runtime = datetime.datetime.now() - template_start_time
            result = pd.DataFrame(
                {
                    'ID': create_model_id(
                        model_str, parameter_dict, transformation_dict
                    ),
                    'Model': model_str,
                    'ModelParameters': json.dumps(parameter_dict),
                    'TransformationParameters': json.dumps(transformation_dict),
                    'Ensemble': ensemble_input,
                    'TransformationRuntime': datetime.timedelta(0),
                    'FitRuntime': fit_runtime,
                    'PredictRuntime': datetime.timedelta(0),
                    'TotalRuntime': fit_runtime,
                    'Exceptions': repr(e),
                    'Runs': 1,
                    'Generation': current_generation,
                    'ValidationRound': validation_round,
                },
                index=[0],
            )
            template_result.model_results = pd.concat(
                [template_result.model_results, result],
                axis=0,
                ignore_index=True,
                sort=False,
            ).reset_index(drop=True)
    if template_result.per_series_mae:
        template_result.per_series_mae = pd.concat(
            template_result.per_series_mae, axis=0
        )
        template_result.per_series_made = pd.concat(
            template_result.per_series_made, axis=0
        )
        template_result.per_series_contour = pd.concat(
            template_result.per_series_contour, axis=0
        )
        template_result.per_series_rmse = pd.concat(
            template_result.per_series_rmse, axis=0
        )
        template_result.per_series_spl = pd.concat(
            template_result.per_series_spl, axis=0
        )
        template_result.per_series_mle = pd.concat(
            template_result.per_series_mle, axis=0
        )
        template_result.per_series_imle = pd.concat(
            template_result.per_series_imle, axis=0
        )
        template_result.per_series_maxe = pd.concat(
            template_result.per_series_maxe, axis=0
        )
        template_result.per_series_oda = pd.concat(
            template_result.per_series_oda, axis=0
        )
        template_result.per_series_mqae = pd.concat(
            template_result.per_series_mqae, axis=0
        )
        template_result.per_series_dwae = pd.concat(
            template_result.per_series_dwae, axis=0
        )
        template_result.per_series_ewmae = pd.concat(
            template_result.per_series_ewmae, axis=0
        )
        template_result.per_series_uwmse = pd.concat(
            template_result.per_series_uwmse, axis=0
        )
        template_result.per_series_smoothness = pd.concat(
            template_result.per_series_smoothness, axis=0
        )
    else:
        template_result.per_series_mae = pd.DataFrame()
        template_result.per_series_made = pd.DataFrame()
        template_result.per_series_contour = pd.DataFrame()
        template_result.per_series_rmse = pd.DataFrame()
        template_result.per_series_spl = pd.DataFrame()
        template_result.per_series_mle = pd.DataFrame()
        template_result.per_series_imle = pd.DataFrame()
        template_result.per_series_maxe = pd.DataFrame()
        template_result.per_series_oda = pd.DataFrame()
        template_result.per_series_mqae = pd.DataFrame()
        template_result.per_series_dwae = pd.DataFrame()
        template_result.per_series_ewmae = pd.DataFrame()
        template_result.per_series_uwmse = pd.DataFrame()
        template_result.per_series_smoothness = pd.DataFrame()
        if verbose > 0 and not template.empty:
            print(f"Generation {current_generation} had all new models fail")
    return template_result


class CustomAutoTS(AutoTS):

    def _run_template(
            self,
            template,
            df_train,
            df_test,
            future_regressor_train,
            future_regressor_test,
            current_weights,
            validation_round=0,
            max_generations="0",
            model_count=None,
            current_generation=0,
            result_file=None,
    ):
        """Get results for one batch of models."""

        # auto-sktime
        caller = inspect.currentframe().f_back
        # Crude hack to enable timeout only during fitting
        use_timeout = caller.f_lineno == 1123 or caller.f_lineno == 1053

        model_count = self.model_count if model_count is None else model_count
        template_result = TemplateWizard(
            template,
            df_train=df_train,
            df_test=df_test,
            weights=current_weights,
            model_count=model_count,
            forecast_length=self.forecast_length,
            frequency=self.frequency,
            prediction_interval=self.prediction_interval,
            no_negatives=self.no_negatives,
            constraint=self.constraint,
            ensemble=self.ensemble,
            future_regressor_train=future_regressor_train,
            future_regressor_forecast=future_regressor_test,
            holiday_country=self.holiday_country,
            startTimeStamps=self.startTimeStamps,
            template_cols=self.template_cols,
            model_interrupt=self.model_interrupt,
            grouping_ids=self.grouping_ids,
            random_seed=self.random_seed,
            verbose=self.verbose,
            max_generations=max_generations,
            n_jobs=self.n_jobs,
            validation_round=validation_round,
            traceback=self.traceback,
            current_model_file=self.current_model_file,
            current_generation=current_generation,
            # auto-sktime
            start_time=self.start_time if use_timeout else None,
            generation_timeout=self.generation_timeout if use_timeout else None,
        )
        if model_count == 0:
            self.model_count += template_result.model_count
        else:
            self.model_count = template_result.model_count
        # capture results from lower-level template run
        template_result.model_results['TotalRuntime'].fillna(
            pd.Timedelta(seconds=60), inplace=True
        )
        # gather results of template run
        self.initial_results = self.initial_results.concat(template_result)
        self.initial_results.model_results['Score'] = generate_score(
            self.initial_results.model_results,
            metric_weighting=self.metric_weighting,
            prediction_interval=self.prediction_interval,
        )
        if result_file is not None:
            self.initial_results.save(result_file)

    def _run_validations(
            self,
            df_wide_numeric,
            num_validations,
            validation_template,
            future_regressor,
            first_validation=True,
    ):
        """Loop through a template for n validation segments."""
        for y in range(num_validations):
            if self.verbose > 0:
                print("Validation Round: {}".format(str(y + 1)))
            # slice the validation data into current validation slice
            current_slice = df_wide_numeric.reindex(self.validation_indexes[(y + 1)])

            # subset series (if used) and take a new train/test split
            if self.subset_flag:
                # mosaic can't handle different cols in each validation
                if any([x in self.mosaic_list for x in self.ensemble]):
                    rand_st = self.random_seed
                else:
                    rand_st = self.random_seed + y + 1
                df_subset = subset_series(
                    current_slice,
                    list((self.weights.get(i)) for i in current_slice.columns),
                    n=self.subset,
                    random_state=rand_st,
                )
                if self.verbose > 1:
                    print(f'{y + 1} subset is of: {df_subset.columns}')
            else:
                df_subset = current_slice
            # subset weighting info
            if not self.weighted:
                current_weights = {x: 1 for x in df_subset.columns}
            else:
                current_weights = {x: self.weights[x] for x in df_subset.columns}

            val_df_train, val_df_test = simple_train_test_split(
                df_subset,
                forecast_length=self.forecast_length,
                min_allowed_train_percent=self.min_allowed_train_percent,
                verbose=self.verbose,
            )
            if first_validation:
                self.validation_train_indexes.append(val_df_train.index)
                self.validation_test_indexes.append(val_df_test.index)
            if self.verbose >= 2:
                print(f'Validation train index is {val_df_train.index}')

            # slice regressor into current validation slices
            if future_regressor is not None:
                val_future_regressor_train = future_regressor.reindex(
                    index=val_df_train.index
                )
                val_future_regressor_test = future_regressor.reindex(
                    index=val_df_test.index
                )
            else:
                val_future_regressor_train = None
                val_future_regressor_test = None

            # force NaN for robustness
            if self.introduce_na or (self.introduce_na is None and self._nan_tail):
                if self.introduce_na:
                    idx = val_df_train.index
                    # make 20% of rows NaN at random
                    val_df_train = val_df_train.sample(
                        frac=0.8, random_state=self.random_seed
                    ).reindex(idx)
                nan_frac = val_df_train.shape[1] / num_validations
                val_df_train.iloc[
                -2:, int(nan_frac * y): int(nan_frac * (y + 1))
                ] = np.nan

            # run validation template on current slice
            self._run_template(
                validation_template,
                val_df_train,
                val_df_test,
                future_regressor_train=val_future_regressor_train,
                future_regressor_test=val_future_regressor_test,
                current_weights=current_weights,
                validation_round=(y + 1),
                max_generations="0",
                model_count=0,
                result_file=None,
            )

        self.validation_results = copy.copy(self.initial_results)
        # aggregate validation results
        self.validation_results = validation_aggregation(
            self.validation_results, df_train=self.df_wide_numeric
        )
