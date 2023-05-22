import pandas as pd
import pyaf


def evaluate_pyaf(y: pd.Series, fh: int):
    y = y.reset_index()

    engine = pyaf.ForecastEngine.cForecastEngine()
    engine.train(iInputDS=y, iTime='index', iSignal='y', iHorizon=7)

    y_pred = engine.forecast(iInputDS=y, iHorizon=fh)
    return y_pred['y_Forecast'].tail(fh), None
