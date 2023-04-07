import argparse
import logging
import os
import sys
from typing import Text, Tuple, Union

import pandas as pd
import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

from src.common import get_logger, get_spark_session


def get_index_range(
    long_covid_silver_standard: DataFrame, microvisits_to_macrovisits: DataFrame
) -> DataFrame:
    spark = get_spark_session()

    long_covid_silver_standard.createOrReplaceTempView("long_covid_silver_standard")
    microvisits_to_macrovisits.createOrReplaceTempView("microvisits_to_macrovisits")

    query = """
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
    FROM long_covid_silver_standard s
    left join microvisits_to_macrovisits v
    on v.person_id = s.person_id
    ) r
    ) f
    where f.rn = 1
    """
    index_range = spark.sql(query)
    return index_range


def get_input_data(config: dict, step: str) -> Tuple[DataFrame, DataFrame]:
    spark = get_spark_session()
    micro_macro = spark.read.csv(
        config["featurize"][step]["micro_to_macro_path"], header=True, inferSchema=True
    )
    micro_macro = with_macro_columns(micro_macro)
    long_covid = spark.read.csv(
        config["featurize"][step]["long_covid_path"], header=True, inferSchema=True
    )
    long_covid = with_pasc_columns(long_covid)
    return (long_covid, micro_macro)


def with_macro_columns(micro_macro: DataFrame) -> DataFrame:
    micro_macro = micro_macro.withColumn("macrovisit_id", F.lit(None).cast("string"))
    micro_macro = micro_macro.withColumn(
        "macrovisit_start_date", F.lit(None).cast(DateType())
    )
    micro_macro = micro_macro.withColumn(
        "macrovisit_end_date", F.lit(None).cast(DateType())
    )
    return micro_macro


def with_pasc_columns(long_covid: DataFrame) -> DataFrame:
    long_covid = long_covid.withColumn(
        "pasc_code_after_four_weeks", F.lit(None).cast("int")
    )
    long_covid = long_covid.withColumn(
        "pasc_code_prior_four_weeks", F.lit(None).cast("int")
    )
    long_covid = long_covid.withColumn("time_to_pasc", F.lit(None).cast("int"))
    return long_covid


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    config_path = args.config

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    long_covid_train, micro_macro_train = get_input_data(config, "train")
    index_range_train = get_index_range(long_covid_train, micro_macro_train)
    index_range_train.write.mode("overwrite").parquet(
        config["featurize"]["train"]["index_range_path"]
    )

    long_covid_test, micro_macro_test = get_input_data(config, "test")
    index_range_test = get_index_range(long_covid_test, micro_macro_test)
    index_range_test.write.mode("overwrite").parquet(
        config["featurize"]["test"]["index_range_path"]
    )
