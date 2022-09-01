import os
import tempfile
from typing import Optional, Any, Dict, Union

import numpy as np
import pandas as pd

from autosktime.automl_common.common.utils.backend import Backend as Backend_, BackendContext
from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.util.singleton import Singleton


class Backend(Backend_):

    def save_targets_ensemble(self, targets: SUPPORTED_Y_TYPES) -> str:
        self._make_internals_directory()

        filepath = self._get_targets_ensemble_filename()

        # Try to open the file without locking it, this will reduce the
        # number of times when we erroneously keep a lock on the ensemble
        # targets file although the process already was killed
        try:
            existing_targets = pd.read_pickle(filepath)
            if existing_targets.shape[0] > targets.shape[0] or (
                    existing_targets.shape == targets.shape and np.allclose(existing_targets, targets)
            ):
                return filepath
        except FileNotFoundError:
            pass

        with tempfile.NamedTemporaryFile("wb", dir=os.path.dirname(filepath), delete=False) as fh_w:
            targets.to_pickle(fh_w)
            tempname = fh_w.name

        os.rename(tempname, filepath)

        return filepath

    def load_targets_ensemble(self) -> SUPPORTED_Y_TYPES:
        filepath = self._get_targets_ensemble_filename()

        with open(filepath, "rb") as fh:
            targets = pd.read_pickle(fh)

        return targets


def create(
        temporary_directory: str,
        output_directory: Optional[str],
        prefix: str,
        delete_tmp_folder_after_terminate: bool = True,
        delete_output_folder_after_terminate: bool = True,
) -> Backend:
    context = BackendContext(
        temporary_directory,
        output_directory,
        delete_tmp_folder_after_terminate,
        delete_output_folder_after_terminate,
        prefix=prefix,
    )
    backend = Backend(context, prefix)

    return backend


ConfigId = int


@Singleton
class ConfigContext:

    def __init__(self):
        self.store: Dict[ConfigId, Dict[str, Any]] = dict()

    def set_config(self, id: ConfigId, config: Dict = None, key: str = None, value: Any = None) -> None:
        if config is None:
            if key is None:
                raise ValueError('Config and key/value pair are both None')
            config = {key: value}
        else:
            if key is not None or value is not None:
                raise ValueError(f'Provide either config or key/value pair, not both')

        if id not in self.store:
            self.store[id] = dict()
        self.store[id].update(config)

    def get_config(self, id: ConfigId, key: str = None) -> Union[Any, Dict[str, Any]]:
        if key is None:
            return self.store[id]
        else:
            return self.store[id].get(key)

    def reset_config(self, id: ConfigId) -> None:
        if id in self.store:
            del self.store[id]
