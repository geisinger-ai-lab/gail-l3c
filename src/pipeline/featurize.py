import argparse
import os
from typing import Text

import pandas as pd
import yaml

from features.smoking_status import get_smoking_status_dataset
from features.vitals import get_vitals_dataset
from src.common import get_logger, get_spark_session
from src.features.medications import get_meds_dataset
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

    spark = get_spark_session()

    # TODO: Actually make the index_range file from raw, instead of using the mock file
    data_path_intermediate = config["featurize"]["data_path_intermediate"]
    index_range_path = os.path.join(data_path_intermediate, "index_range.csv")
    index_range = spark.read.csv(index_range_path, header=True, inferSchema=True)

    # Load the concept sets file as a spark dataframe
    concept_set_members_path = config["featurize"]["concept_set_members"]
    concept_set_members = spark.read.csv(concept_set_members_path, header=True)

    # Load OMOP data as spark dataframes
    data_path_raw = config["featurize"]["data_path_raw"]
    drug_exposure_path = os.path.join(data_path_raw, "drug_exposure.csv")
    drug_exposure = spark.read.csv(drug_exposure_path, header=True, inferSchema=True)

    measurement_path = os.path.join(data_path_raw, "measurement.csv")
    measurement = spark.read.csv(measurement_path, header=True, inferSchema=True)

    observation_path = os.path.join(data_path_raw, "observation.csv")
    observation = spark.read.csv(observation_path, header=True, inferSchema=True)

    # Call src.featurize functions to get datasets
    meds_features = get_meds_dataset(concept_set_members, drug_exposure, index_range)
    vitals_features = get_vitals_dataset(concept_set_members, measurement, index_range)
    smoking_features = get_smoking_status_dataset(concept_set_members, observation)

    utilization = get_utilization()

    logger.info(f"Utilization columns: {[c for c in utilization.columns]}")

    # TODO: reconcile expected columns with what is missing/unavailable in synpuf

    # TODO: merge features into a single dataframe (pandas?)
    # TODO: implement missing value imputation functions
    # TODO: save final person-by-feature dataframe to data/featurized/

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
