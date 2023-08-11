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
        iSignal='y',
        iHorizon=fh,
        iExogenousData=(X_train, set(X_train.columns) - {'index', 'series'}) if X_train is not None else None
    )
    print('#########')
    engine.getModelInfo()
    print('#########')

    y_pred = engine.forecast(iInputDS=y, iHorizon=fh)
    y_pred_ints = pd.DataFrame(
        y_pred[['y_Forecast_Lower_Bound', 'y_Forecast_Upper_Bound']].values,
        columns=pd.MultiIndex.from_tuples([('Coverage', 0.5, 'lower'), ('Coverage', 0.5, 'upper')]),
        index=y_pred.index
    ).tail(fh)

    return y_pred['y_Forecast'].tail(fh), y_pred_ints
