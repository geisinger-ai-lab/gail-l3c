"""
Home for global functions used across the codebase 
"""

import logging
import os
import sys
from typing import Text, Union

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

## Vitals Features:
feat_vitals = [
    "bp_diastolic_before",
    "bp_systolic_before",
    "heart_rate_before",
    "resp_rate_before",
    "spo2_before",
    "bp_diastolic_during",
    "bp_systolic_during",
    "heart_rate_during",
    "resp_rate_during",
    "spo2_during",
    "bp_diastolic_after",
    "bp_systolic_after",
    "heart_rate_after",
    "resp_rate_after",
    "spo2_after",
]

## Smoking Features:
feat_smoke = ["smoker"]

## Medication Count Features:
feat_meds = [
    "anticoagulants_before",
    "asthma_drugs_before",
    "antibiotics_during",
    "antivirals_during",
    "corticosteroids_during",
    "iv_immunoglobulin_during",
    "lopinavir_during",
    "paxlovid_during",
    "remdesivir_during",
    "anticoagulants_after",
    "asthma_drugs_after",
]


def get_index_range(index_range_path):
    """
    -- COVID Index Date Range (a30509ba-7358-4b35-a294-19ef1c566c9b): v2
    -- person_id, visit_occurrence_id, covid_index, visit_start_date, visit_end_date, macrovisit_start_date, macrovisit_end_date, covid_index_start, covid_index_end, silver_standard.*

    -- if no visit, use covid index for all dates

    select f.person_id,
    f.visit_occurrence_id,
    f.macrovisit_id,
    f.covid_index,
    f.visit_start_date,
    f.visit_end_date,
    f.macrovisit_start_date,
    f.macrovisit_end_date,
    f.index_start_date,
    f.index_end_date,
    f.pasc_code_after_four_weeks,
    f.pasc_code_prior_four_weeks,
    f.time_to_pasc  from
    (
    select r.person_id,
    r.visit_occurrence_id,
    r.macrovisit_id,
    r.covid_index,
    r.visit_start_date,
    r.visit_end_date,
    r.macrovisit_start_date,
    r.macrovisit_end_date,
    r.index_start_date,
    r.index_end_date,
    r.pasc_code_after_four_weeks,
    r.pasc_code_prior_four_weeks,
    r.time_to_pasc,
    case when abs_visit_to_covid_diff is not null then
    row_number() over(partition by person_id order by abs_visit_to_covid_diff)
    else 1 end rn from
    (
    SELECT s.person_id,
    v.visit_occurrence_id,
    v.macrovisit_id,
    s.covid_index,
    v.visit_start_date,
    v.visit_end_date,
    v.macrovisit_start_date,
    v.macrovisit_end_date,
    coalesce(v.macrovisit_start_date, v.visit_start_date, s.covid_index) index_start_date,
    coalesce(v.macrovisit_end_date, v.visit_end_date, s.covid_index) index_end_date,
    abs(datediff(v.visit_start_date, s.covid_index)) abs_visit_to_covid_diff,
    s.pasc_code_after_four_weeks,
    s.pasc_code_prior_four_weeks,
    s.time_to_pasc
    FROM Long_COVID_Silver_Standard s
    left join microvisits_to_macrovisits_merge v
    on v.person_id = s.person_id
    ) r
    ) f
    where f.rn = 1
    """
    if os.path.exists(index_range_path):
        df = pd.read_csv(index_range_path)
    else:
        columns = [
            "person_id",
            "visit_occurrence_id",
            "macrovisit_id",
            "covid_index",
            "visit_start_date",
            "visit_end_date",
            "macrovisit_start_date",
            "macrovisit_end_date",
            "index_start_date",
            "index_end_date",
            "pasc_code_after_four_weeks",
            "pasc_code_prior_four_weeks",
            "time_to_pasc",
        ]
        df = pd.DataFrame(columns=columns)
    return df


def rename_cols(df, prefix="", suffix=""):
    """
    Helper function to rename columns by adding a prefix or a suffix
    """
    index_cols = ["person_id", "before_or_after_index"]
    select_list = [
        col(col_name).alias(prefix + col_name + suffix)
        if col_name not in index_cols
        else col(col_name)
        for col_name in df.columns
    ]
    df = df.select(select_list).drop(col("before_or_after_index"))
    return df


def get_console_handler() -> logging.StreamHandler:
    """Get console handler.
    Returns:
        logging.StreamHandler which logs into stdout
    """

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    )
    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(
    name: Text = __name__, log_level: Union[Text, int] = logging.DEBUG
) -> logging.Logger:
    """Get logger.
    Args:
        name {Text}: logger name
        log_level {Text or int}: logging level; can be string name or integer value
    Returns:
        logging.Logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate outputs in Jypyter Notebook
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_console_handler())
    logger.propagate = False

    return logger


def get_spark_session():
    spark = SparkSession.builder.appName("L3C").getOrCreate()
    return spark
