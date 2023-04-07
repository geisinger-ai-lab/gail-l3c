import argparse
import os
from textwrap import fill
from typing import Text

import pandas as pd
import yaml
from pyspark.sql.functions import col

from src.common import (
    add_missing_cols,
    feat_demo,
    feat_dxct,
    feat_meas_after,
    feat_meas_before,
    feat_meas_during,
    feat_meds,
    feat_proc,
    feat_smoke,
    feat_utl,
    feat_vitals,
    get_logger,
    get_spark_session,
    label,
)
from src.features.demographics import person_demographics
from src.features.diagnoses import get_diagnoses
from src.features.labs import get_labs
from src.features.medications import get_meds_dataset
from src.features.procedures import get_procedure_dataset
from src.features.smoking_status import get_smoking_status_dataset
from src.features.utilization import get_utilization
from src.features.vitals import get_vitals_dataset

spark = get_spark_session()


def load_omop_data(config):

    # Load OMOP data as spark dataframes
    data_path_raw = config["featurize"]["data_path_raw"]
    drug_exposure_path = os.path.join(data_path_raw, "drug_exposure.csv")
    drug_exposure = spark.read.csv(drug_exposure_path, header=True, inferSchema=True)

    measurement_path = os.path.join(data_path_raw, "measurement.csv")
    measurement = spark.read.csv(measurement_path, header=True, inferSchema=True)

    observation_path = os.path.join(data_path_raw, "observation.csv")
    observation = spark.read.csv(observation_path, header=True, inferSchema=True)

    person_path = os.path.join(data_path_raw, "person.csv")
    person = spark.read.csv(person_path, header=True, inferSchema=True)

    condition_occurrence_path = os.path.join(data_path_raw, "condition_occurrence.csv")
    condition_occurrence = spark.read.csv(condition_occurrence_path, header=True, inferSchema=True)

    procedure_occurrence_path = os.path.join(data_path_raw, "procedure_occurrence.csv")
    procedure_occurrence = spark.read.csv(procedure_occurrence_path, header=True, inferSchema=True)

    return (
        condition_occurrence,
        drug_exposure,
        measurement,
        observation,
        person,
        procedure_occurrence,
    )


def load_concept_set_members(config):

    # Load the concept sets file as a spark dataframe
    concept_set_members_path = config["featurize"]["concept_set_members"]
    concept_set_members = spark.read.csv(concept_set_members_path, header=True)

    return concept_set_members


def load_intermediate_data(config):
    # TODO: Actually make the index_range file from raw, instead of using the mock file
    data_path_intermediate = config["featurize"]["data_path_intermediate"]
    index_range_path = os.path.join(data_path_intermediate, "index_range.csv")
    index_range = spark.read.csv(index_range_path, header=True, inferSchema=True)

    return index_range


def missing_value_imputation(df_feat):

    ## Addressing the missing values in different cases:

    # Imputation with value '0': Diagnosis Count, Procedures, Medication Count, Utilization count
    df = df_feat.na.fill(0, subset=feat_dxct + feat_proc + feat_meds + feat_utl)

    # Imputing with -1: Smoking Status,  Measurements Values
    df = df.na.fill(
        -1, subset=feat_smoke + feat_vitals + feat_meas_after + feat_meas_before + feat_meas_during
    )

    return df


def featurize(config_path: Text) -> None:
    """Featurize data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("FEATURIZE", log_level=config["common"]["log_level"])

    logger.info("Featurizing...")

    # Load data as Spark
    concept_set_members = load_concept_set_members(config)
    (
        condition_occurrence,
        drug_exposure,
        measurement,
        observation,
        person,
        procedure_occurrence,
    ) = load_omop_data(config)
    index_range = load_intermediate_data(config)

    # Call src.featurize functions to get each features' dataframe
    demographics_features = person_demographics(person)
    diagnoses_features = get_diagnoses(condition_occurrence, concept_set_members)
    labs_features = get_labs()  # TODO <-- placeholder
    medications_features = get_meds_dataset(concept_set_members, drug_exposure, index_range)
    smoking_features = get_smoking_status_dataset(concept_set_members, observation)
    procedures_features = get_procedure_dataset(
        concept_set_members, procedure_occurrence, index_range
    )
    utilization_features = get_utilization()  # TODO <-- placeholder
    vitals_features = get_vitals_dataset(concept_set_members, measurement, index_range)
    label_df = index_range.select(["person_id", label])

    logger.info(f"Utilization columns: {[c for c in utilization_features.columns]}")

    # Add empty columns for features that may be missing
    demographics_features = add_missing_cols(df=demographics_features, col_list=feat_demo)
    diagnoses_features = add_missing_cols(df=diagnoses_features, col_list=feat_dxct)
    # labs_features      = add_missing_cols(df=labs_features, col_list=feat_demo, fill_val=-1)
    medications_features = add_missing_cols(df=medications_features, col_list=feat_meds)
    smoking_features = add_missing_cols(df=smoking_features, col_list=feat_smoke, fill_val=-1)
    procedures_features = add_missing_cols(df=procedures_features, col_list=feat_proc)
    utilization_features = add_missing_cols(df=utilization_features, col_list=feat_utl)
    vitals_features = add_missing_cols(df=vitals_features, col_list=feat_vitals, fill_val=-1)

    # Join all of the features together
    featurized_df = (
        demographics_features.join(diagnoses_features, on="person_id", how="left")
        .join(labs_features, on="person_id", how="left")
        .join(medications_features, on="person_id", how="left")
        .join(smoking_features, on="person_id", how="left")
        .join(procedures_features, on="person_id", how="left")
        .join(utilization_features, on="person_id", how="left")
        .join(vitals_features, on="person_id", how="left")
        .join(label_df, on="person_id", how="left")
    )

    logger.info(
        f"Featurized dataframe shape: ({featurized_df.count()}, {len(featurized_df.columns)})"
    )

    # Fill missing values
    featurized_df = missing_value_imputation(featurized_df)

    logger.info(f"Featurized dataframe schema: {featurized_df.printSchema()}")

    ## Train / test split
    data_path_featurized = config["featurize"]["data_path_featurized"]
    if not os.path.exists(data_path_featurized):
        os.mkdir(data_path_featurized)
    if config["featurize"].get("test_set_pct"):
        test_set_pct = config["featurize"]["test_set_pct"]
        assert test_set_pct < 1
        assert test_set_pct > 0
        random_seed = config["featurize"].get("random_seed")
        train_df, test_df = featurized_df.randomSplit(
            [1 - test_set_pct, test_set_pct], seed=random_seed
        )
        train_df_path = os.path.join(data_path_featurized, "featurized_data_train.csv")
        logger.info(f"Writing training set to {train_df_path}")
        train_df.toPandas().to_csv(train_df_path, index=None)

        test_df_path = os.path.join(data_path_featurized, "featurized_data_test.csv")
        logger.info(f"Writing test set to {train_df_path}")
        test_df.toPandas().to_csv(test_df_path, index=None)

    else:
        # If test_set_pct is not specified, just write the full featurized_df
        featurized_df_path = os.path.join(data_path_featurized, "featurized_data.csv")
        logger.info(f"Writing featurized data set to {featurized_df_path}")
        featurized_df.toPandas().to_csv(featurized_df_path, index=None)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)
