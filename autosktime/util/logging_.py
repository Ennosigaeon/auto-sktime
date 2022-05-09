import logging.config
import os
from typing import Optional, Dict

import yaml


def setup_logger(logging_config: Optional[Dict] = None) -> None:
    if logging_config is None:
        with open(os.path.join(os.path.dirname(__file__), 'logging.yaml'), 'r') as fh:
            logging_config = yaml.safe_load(fh)

    logging.config.dictConfig(logging_config)
