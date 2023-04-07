import argparse
from typing import List, Text, Tuple, Union
import os

import pandas as pd
import yaml
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common import get_spark_session
from src.features.index_range import get_index_range, get_micro_macro_long_covid

## Utilization
CAP_LOS_VALUES = False
LOS_MAX = 365


def get_utilization(config: dict) -> DataFrame:
    (
        microvisits_to_macrovisits,
        concept_set_members,
        procedure_occurrence,
        condition_occurrence,
        observation,
        long_covid_silver_standard,
        index_range,
    ) = get_input_data(config)

    utilization = add_icu(
        microvisits_to_macrovisits,
        concept_set_members,
        procedure_occurrence,
        condition_occurrence,
        observation,
    )

    utilization = add_los_covid_index(utilization, long_covid_silver_standard)

    utilization = cap_los_outliers(utilization)

    ed_ip_op = add_ed_ip_op(utilization)

    index_concept_names = index_visit_concept_name(index_range, ed_ip_op)

    with_los = los_stats(ed_ip_op)

    with_before = before_index_visit_name_counts(ed_ip_op, index_range)

    with_during = during_index_visit_name_counts(ed_ip_op, index_range)

    with_after = after_index_visit_name_counts(ed_ip_op, index_range)

    final_util = add_final_columns(
        index_concept_names, with_los, with_before, with_during, with_after
    )

    imputed_final = impute_features(final_util)

    return imputed_final


def with_concept_name(
    df, domain_df, icu_concepts, concept_id_column, concept_name_column
):
    df = df.join(domain_df, "visit_occurrence_id", how="left")

    df = df.join(
        icu_concepts, df[concept_id_column] == icu_concepts["concept_id"], how="left"
    )

    df = (
        df.withColumnRenamed("concept_name", concept_name_column)
        .drop("concept_id")
        .distinct()
    )

    df = df.withColumn(
        "rn",
        F.row_number().over(
            Window.partitionBy("visit_occurrence_id").orderBy(
                F.desc(concept_name_column)
            )
        ),
    )

    df = df.filter(df["rn"] == 1)

    df = df.drop("rn")

    return df


# Add_ICU (d5691458-1e67-4887-a68f-4cfbd4753295): v1


def add_icu(
    microvisits_to_macrovisits,
    concept_set_members,
    procedure_occurrence,
    condition_occurrence,
    observation,
):
    icu_codeset_id = 469361388

    icu_concepts = concept_set_members.filter(F.col("codeset_id") == icu_codeset_id).select(
        "concept_id", "concept_name"
    )

    procedures_df = procedure_occurrence[
        ["visit_occurrence_id", "procedure_concept_id"]
    ]
    condition_df = condition_occurrence[["visit_occurrence_id", "condition_concept_id"]]
    observation_df = observation[["visit_occurrence_id", "observation_concept_id"]]

    df = microvisits_to_macrovisits

    df = with_concept_name(
        df,
        procedures_df,
        icu_concepts,
        "procedure_concept_id",
        "procedure_concept_name",
    )
    df = with_concept_name(
        df, condition_df, icu_concepts, "condition_concept_id", "condition_concept_name"
    )
    df = with_concept_name(
        df,
        observation_df,
        icu_concepts,
        "observation_concept_id",
        "observation_concept_name",
    )

    df = df.withColumn(
        "is_icu",
        F.when(
            F.coalesce(
                df["procedure_concept_name"],
                df["condition_concept_name"],
                df["observation_concept_name"],
            ).isNotNull(),
            1,
        ).otherwise(0),
    )

    return df


def add_los_covid_index(
    add_icu: DataFrame, long_covid_silver_standard: DataFrame
) -> DataFrame:
    # -- Add LOS and COVID Index (016fc630-b7c6-405e-b4a8-7c5bbb03dfef): v1
    spark = get_spark_session()

    long_covid_silver_standard.createOrReplaceTempView("long_covid_silver_standard")
    add_icu.createOrReplaceTempView("add_icu")

    query = """SELECT icu.*, s.covid_index, 
    coalesce(icu.macrovisit_start_date, icu.visit_start_date) stay_start_date,
    coalesce(icu.macrovisit_end_date, icu.visit_end_date) stay_end_date, 
    case when 
    coalesce(icu.macrovisit_end_date, icu.visit_end_date) is not null then
    datediff(coalesce(icu.macrovisit_end_date, icu.visit_end_date), coalesce(icu.macrovisit_start_date, icu.visit_start_date)) 
    else 0 end los
    FROM add_icu icu 
    left join long_covid_silver_standard s on icu.person_id = s.person_id
    """
    los_covid_index = spark.sql(query)
    return los_covid_index


def cap_los_outliers(add_los_and_index):
    df = add_los_and_index

    if CAP_LOS_VALUES:
        df = df.withColumn(
            "los_mod",
            F.when(F.col("los") > LOS_MAX, 0).when(F.col("los") < 0, 0).otherwise(F.col("los")),
        )
        df = df.drop("los").withColumnRenamed("los_mod", "los")

    return df


def add_ed_ip_op(los: DataFrame) -> DataFrame:
    spark = get_spark_session()

    los.createOrReplaceTempView("los")

    query = """SELECT *, 
    case when visit_concept_name like '%Emergency%' then 1 else 0 end is_ed,
    case when visit_concept_name like '%Inpatient%' then 1 else 0 end is_ip,
    case when visit_concept_name like '%Tele%' then 1 else 0 end is_tele,
    case when visit_concept_name not like '%Emergency%' 
        and visit_concept_name not like '%Inpatient%' 
        and visit_concept_name not like '%Tele%' then 1 else 0 end is_op
    FROM los
    """

    df = spark.sql(query)
    return df


def before_index_visit_name_counts(ed_ip_op, index_range):
    idx_df = index_range.select("person_id", "index_start_date", "index_end_date")
    df = ed_ip_op.join(idx_df, "person_id", how="left")
    before_df = df.where(
        F.coalesce(F.col("visit_end_date"), F.col("visit_start_date")) < F.col("index_start_date")
    )

    counts_df = before_df.groupBy("person_id").agg(
        F.sum("is_ed").alias("before_ed_cnt"),
        F.sum("is_ip").alias("before_ip_cnt"),
        F.sum("is_op").alias("before_op_cnt"),
        F.sum("is_tele").alias("before_tele_cnt"),
    )

    return counts_df


def during_index_visit_name_counts(ed_ip_op, index_range):
    idx_df = index_range.select("person_id", "index_start_date", "index_end_date")
    df = ed_ip_op.join(idx_df, "person_id", how="left")

    during_df = df.where(
        (F.col("visit_start_date") >= F.col("index_start_date"))
        & (
            F.coalesce(F.col("visit_end_date"), F.col("visit_start_date"))
            <= F.col("index_end_date")
        )
    )

    counts_df = during_df.groupBy("person_id").agg(
        F.sum("is_ed").alias("during_ed_cnt"),
        F.sum("is_ip").alias("during_ip_cnt"),
        F.sum("is_op").alias("during_op_cnt"),
        F.sum("is_tele").alias("during_tele_cnt"),
    )

    return counts_df


def after_index_visit_name_counts(ed_ip_op, index_range):
    idx_df = index_range.select("person_id", "index_start_date", "index_end_date")
    df = ed_ip_op.join(idx_df, "person_id", how="left")
    during_df = df.where(F.col("visit_start_date") > F.col("index_end_date"))

    counts_df = during_df.groupBy("person_id").agg(
        F.sum("is_ed").alias("after_ed_cnt"),
        F.sum("is_ip").alias("after_ip_cnt"),
        F.sum("is_op").alias("after_op_cnt"),
        F.sum("is_tele").alias("after_tele_cnt"),
    )

    return counts_df


def index_visit_concept_name(index_range: DataFrame, ed_ip_op: DataFrame) -> DataFrame:
    spark = get_spark_session()

    index_range.createOrReplaceTempView("index_range")
    ed_ip_op.createOrReplaceTempView("ed_ip_op")

    query = """select 
    c.person_id, 
    case when any(c.visit_concept_name like '%Emergency%') then 1 else 0 end is_index_ed,
    case when any(c.visit_concept_name like '%Inpatient%') then 1 else 0 end is_index_ip,
    case when any(c.visit_concept_name like '%Tele%') then 1 else 0 end is_index_tele,
    case when (any(c.visit_concept_name not like '%Emergency%')
            and any(c.visit_concept_name not like '%Inpatient%')
            and any(c.visit_concept_name not like '%Tele%')) then 1 else 0 end is_index_op
    from
    (SELECT distinct i.person_id, 
    v.visit_concept_name
    FROM index_range i
    left join ed_ip_op v on i.person_id = v.person_id) c
    group by c.person_id"""

    return spark.sql(query)


def los_stats(ed_ip_op: DataFrame) -> DataFrame:
    spark = get_spark_session()

    ed_ip_op.createOrReplaceTempView("ed_ip_op")

    query = """
    SELECT
        overall.person_id,
        overall.avg_los,
        CASE
            WHEN icu.avg_icu_los IS NOT NULL THEN icu.avg_icu_los
            ELSE 0
        END avg_icu_los
    FROM (SELECT
            person_id,
            AVG(los) avg_los
        FROM (SELECT DISTINCT
                person_id,
                stay_start_date,
                stay_end_date,
                los
            FROM ed_ip_op) o
        GROUP BY person_id) overall
    LEFT JOIN (SELECT
            person_id,
            AVG(los) avg_icu_los
        FROM (SELECT DISTINCT
                person_id,
                stay_start_date,
                stay_end_date,
                los
            FROM ed_ip_op
            WHERE is_icu = 1) o
        GROUP BY person_id) icu
        ON icu.person_id = overall.person_id
    """
    return spark.sql(query)


def add_final_columns(
    index_visit_concept_name,
    los_stats,
    before_index_visit_name_counts,
    during_index_visit_name_counts,
    after_index_visit_name_counts,
):
    spark = get_spark_session()

    index_visit_concept_name.createOrReplaceTempView("index_visit_concept_name")
    los_stats.createOrReplaceTempView("los_stats")
    before_index_visit_name_counts.createOrReplaceTempView(
        "before_index_visit_name_counts"
    )
    during_index_visit_name_counts.createOrReplaceTempView(
        "during_index_visit_name_counts"
    )
    after_index_visit_name_counts.createOrReplaceTempView(
        "after_index_visit_name_counts"
    )

    query = """
    SELECT
        i.person_id,
        i.is_index_ed,
        i.is_index_ip,
        i.is_index_tele,
        i.is_index_op,
        l.avg_los,
        l.avg_icu_los,
        b.before_ed_cnt,
        b.before_ip_cnt,
        b.before_op_cnt,
        d.during_ed_cnt,
        d.during_ip_cnt,
        d.during_op_cnt,
        a.after_ed_cnt,
        a.after_ip_cnt,
        a.after_op_cnt
    FROM index_visit_concept_name i
    LEFT JOIN los_stats l
        ON i.person_id = l.person_id
    LEFT JOIN before_index_visit_name_counts b
        ON i.person_id = b.person_id
    LEFT JOIN during_index_visit_name_counts d
        ON i.person_id = d.person_id
    LEFT JOIN after_index_visit_name_counts a
        ON i.person_id = a.person_id
    """

    return spark.sql(query)


def impute_features(final_df):
    df = final_df
    subset_fill_0 = [
        "is_index_ed",
        "is_index_ip",
        "is_index_tele",
        "is_index_op",
        "before_ed_cnt",
        "before_ip_cnt",
        "before_op_cnt",
        "during_ed_cnt",
        "during_ip_cnt",
        "during_op_cnt",
        "after_ed_cnt",
        "after_ip_cnt",
        "after_op_cnt",
    ]
    subset_fill_neg_1 = ["avg_los", "avg_icu_los"]
    df = df.fillna(0, subset=subset_fill_0)
    df = df.fillna(-1, subset=subset_fill_neg_1)
    return df


def get_input_data(config: dict) -> Tuple[DataFrame]:
    spark = get_spark_session()
    data_path_raw = config["featurize"]["data_path_raw"]
    
    concept_set_members = spark.read.csv(
        config["featurize"]["concept_set_members"], header=True, inferSchema=True
    )
    procedure_occurrence = spark.read.csv(
        os.path.join(data_path_raw, "procedure_occurrence.csv"), header=True, inferSchema=True
    )
    condition_occurrence = spark.read.csv(
        os.path.join(data_path_raw, "condition_occurrence.csv"), header=True, inferSchema=True
    )
    observation = spark.read.csv(
        os.path.join(data_path_raw, "observation.csv"), header=True, inferSchema=True
    )
    (long_covid_silver_standard, microvisits_to_macrovisits) = get_micro_macro_long_covid(config)
    index_range = get_index_range(config)
    return (
        microvisits_to_macrovisits,
        concept_set_members,
        procedure_occurrence,
        condition_occurrence,
        observation,
        long_covid_silver_standard,
        index_range,
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    config_path = args.config

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    df = get_utilization(config)
    df.show()
