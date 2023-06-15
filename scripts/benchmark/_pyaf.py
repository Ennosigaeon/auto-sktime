import pandas as pd
import pyaf


def evaluate_pyaf(y: pd.Series, fh: int, max_duration: int, name: str):
    y = y.reset_index()

    engine = pyaf.ForecastEngine.cForecastEngine()
    engine.mOptions.mNbCores = 1
    engine.train(iInputDS=y, iTime='index', iSignal='y', iHorizon=fh)
    print(engine.getModelInfo())

    y_pred = engine.forecast(iInputDS=y, iHorizon=fh)
    y_pred_ints = pd.DataFrame(
        y_pred[['y_Forecast_Lower_Bound', 'y_Forecast_Upper_Bound']].values,
        columns=pd.MultiIndex.from_tuples([('Coverage', 0.5, 'lower'), ('Coverage', 0.5, 'upper')]),
        index=y_pred.index
    ).tail(fh)

    return y_pred['y_Forecast'].tail(fh), y_pred_ints
