import argparse
from typing import Text

import pandas as pd
import yaml

from src.common import get_logger

from src.features.utilization import get_utilization


def featurize(config_path: Text) -> None:
    """Featurize data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("FEATURIZE", log_level=config["common"]["log_level"])

    logger.info("Featurizing...")

    utilization = get_utilization()

    logger.info(f"Utilization columns: {[c for c in utilization.columns]}")
    """

    index_range 

    each feature file's main public funtion

    the rest of the merge/inputation steps

    save somewhere

    (repeat with paths for training data, testing data)

    """


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)
