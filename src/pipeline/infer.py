import argparse
from typing import Text

import pandas as pd
import yaml

from src.common import get_logger


def infer(config_path: Text) -> None:
    """Infer
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("INFER", log_level=config["common"]["log_level"])

    logger.info("Generating inferences...")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    infer(config_path=args.config)
