from typing import Optional

import pandas as pd


def evaluate_pyaf(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int
):
    import pyaf

    y = y.reset_index()
    if X_train is not None:
        X_train = X_train.reset_index()
    if X_test is not None:
        X_test = X_test.reset_index()

    engine = pyaf.ForecastEngine.cForecastEngine()
    engine.mOptions.mNbCores = 1
    engine.train(
        iInputDS=y,
        iTime='index',
        iSignal='y' if 'y' in y.columns else list(set(y.columns) - {'index', 'series'}),
        iHorizon=fh,
        iExogenousData=(X_train, set(X_train.columns) - {'index', 'series'}) if X_train is not None else None
    )
    print('#########')
    engine.getModelInfo()
    print('#########')

    y_pred = engine.forecast(iInputDS=y, iHorizon=fh)

    y_pred_ints = pd.DataFrame(
        y_pred[
            [f'{col}_Forecast_Lower_Bound' for col in y.columns if col not in ('index', 'series')] +
            [f'{col}_Forecast_Upper_Bound' for col in y.columns if col not in ('index', 'series')]
            ].values,
        columns=pd.MultiIndex.from_tuples(
            [(f'Coverage_{col}', 0.5, 'lower') for col in y.columns if col not in ('index', 'series')] +
            [(f'Coverage_{col}', 0.5, 'upper') for col in y.columns if col not in ('index', 'series')]
        ),
        index=y_pred.index
    ).tail(fh)
    return y_pred[[f'{col}_Forecast' for col in y.columns if col not in ('index', 'series')]].tail(fh), y_pred_ints
