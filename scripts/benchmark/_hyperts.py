import numpy as np
import pandas as pd
from hyperts import make_experiment
from hyperts.framework.compete import TSPipeline


def evaluate_hyperts(y: pd.Series, fh: int):
    y = y.reset_index()

    model: TSPipeline = make_experiment(y, task='univariate-forecast', timestamp='index').run()

    y_pred = pd.DataFrame(np.arange(y['index'].max(), y['index'].max() + fh), columns=['index'])
    y_pred = model.predict(y_pred)

    return y_pred['y'], None
