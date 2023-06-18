import numpy as np
import pandas as pd
from hyperts import make_experiment
from hyperts.framework.compete import TSPipeline

from scripts.benchmark.util import generate_fh


def evaluate_hyperts(y: pd.Series, fh: int, max_duration: int, name: str):
    fh = pd.DataFrame(generate_fh(y.index, fh).to_timestamp(), columns=['index'])
    fh.index += y.shape[0]
    y = y.reset_index()

    model: TSPipeline = make_experiment(
        y,
        task='univariate-forecast',
        timestamp='index',
        max_trials=500000,
        early_stopping_rounds=500000,
        early_stopping_time_limit=max_duration) \
        .run()

    y_pred = model.predict(fh).set_index('index')
    y_pred_ints = pd.DataFrame(
        np.tile(y_pred.values, 2),
        columns=pd.MultiIndex.from_tuples([('Coverage', 0.5, 'lower'), ('Coverage', 0.5, 'upper')]),
        index=y_pred.index
    )

    return y_pred['y'], y_pred_ints
