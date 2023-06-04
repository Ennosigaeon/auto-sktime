import pandas as pd
from autots import AutoTS


def evaluate_autots(y: pd.Series, fh: int):
    df = y.to_frame()

    model = AutoTS(
        forecast_length=fh,
        prediction_interval=0.5,
        max_generations=1
    )
    model = model.fit(df)
    output = model.predict()

    print(model)

    predictions = output.forecast
    y_pred_ints = pd.DataFrame(
        output.upper_forecast.join(output.lower_forecast, rsuffix='r').values,
        columns=pd.MultiIndex.from_tuples([('Coverage', 0.5, 'lower'), ('Coverage', 0.5, 'upper')]),
        index=predictions.index
    )

    return predictions, y_pred_ints
