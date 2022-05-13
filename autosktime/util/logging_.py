import os
from typing import Optional, Dict, Any

import yaml

from autosktime.automl_common.common.utils.logging_ import setup_logger as setup_logger_


def setup_logger(
        output_dir: str,
        filename: Optional[str] = None,
        logging_config: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    if logging_config is None:
        with open(os.path.join(os.path.dirname(__file__), 'logging.yaml'), 'r') as fh:
            logging_config = yaml.safe_load(fh)

    setup_logger_(output_dir=output_dir, filename=filename, logging_config=logging_config)
