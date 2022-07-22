import pandas as pd


def resolve_index(y_index: pd.Index) -> pd.Index:
    if isinstance(y_index, pd.MultiIndex):
        # Use place-holder for panel data as actual forecasting horizons are determined on-demand later on
        y_index = y_index.remove_unused_levels().levels[-1]
    return y_index
