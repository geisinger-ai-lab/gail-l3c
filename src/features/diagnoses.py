# Condition_occurence
"""
function to load diagnosis data

"""
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# from pyspark.sql import types, functions as T,F
from pyspark.sql.functions import when

from src.common import get_spark_session, rename_cols

spark = get_spark_session()

"""
@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ed6feda8-c72f-4a3e-8c6e-a97c6695d0f2"),
    condition_occurrence_test=Input(rid="ri.foundry.main.dataset.3e01546f-f110-4c67-a6db-9063d2939a74"),
    condition_occurrence_train=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2")
)
SELECT 
person_id,
condition_occurrence_id,
condition_end_date,
condition_end_datetime,
condition_start_date,
condition_start_datetime,
data_partner_id,
provider_id,
stop_reason,
visit_detail_id,
visit_occurrence_id,
condition_source_value,
condition_status_source_value,
condition_concept_id,
condition_source_concept_id,
condition_status_concept_id,
condition_type_concept_id,
condition_concept_name,
condition_source_concept_name,
condition_status_concept_name,
condition_type_concept_name
FROM condition_occurrence_train
UNION ALL
SELECT 
person_id,
condition_occurrence_id,
condition_end_date,
condition_end_datetime,
condition_start_date,
condition_start_datetime,
data_partner_id,
provider_id,
stop_reason,
visit_detail_id,
visit_occurrence_id,
condition_source_value,
condition_status_source_value,
condition_concept_id,
condition_source_concept_id,
condition_status_concept_id,
condition_type_concept_id,
condition_concept_name,
condition_source_concept_name,
condition_status_concept_name,
condition_type_concept_name
FROM condition_occurrence_test
"""

# COHORT_DIAGNOSIS_CURATED
def cohort_diagnosis_curated(condition_occurrence, concept_set_members):
    condition_occurrence.createOrReplaceTempView("condition_occurrence")
    concept_set_members.createOrReplaceTempView("concept_set_members")

    cond_sql = """SELECT *
        FROM condition_occurrence as cond 
        INNER JOIN concept_set_members as set_cond 
            ON cond.condition_concept_id = set_cond.concept_id
        WHERE set_cond.codeset_id IN ('834391873', '882775108', '18918743', '248468138', '581513221', '628969102', '602584947','33199070')
        ORDER BY cond.person_id 

    --  WHERE set_cond.concept_set_name IN ('HYPERTENSION', 'HEART FAILURE', 'DIABETES COMPLICATED', 'DIABETES UNCOMPLICATED', 'OBESITY', 'TOBACCO SMOKER', '[N3C][GAIL]asthma','[L3C][GAIL] COPD')
    --  ORDER BY cond.person_id 

    """

    return spark.sql(cond_sql)


# from pyspark.sql.functions import expr


def cohort_dx_ct_features(COHORT_DIAGNOSIS_CURATED):
    ## Initialize the input with the table:
    df = COHORT_DIAGNOSIS_CURATED

    ## Rename measurement concept name for future use:
    name_dict = {
        "DIABETES COMPLICATED": "diabetes_complicated",
        "DIABETES UNCOMPLICATED": "diabetes_uncomplicated",
        "[VSAC] Asthma ": "asthma",
        "[L3C][GAIL] COPD": "copd",
        "HYPERTENSION": "hypertension",
        "HEART FAILURE": "heart_failure",
        "OBESITY": "obesity",
        "TOBACCO SMOKER": "tobacco_smoker",
    }
    df = df.replace(to_replace=name_dict, subset=["concept_set_name"])

    ## Pivoting the table and gathering the related diagnosis count:
    pivotDF = (
        df.groupBy(["person_id"])
        .pivot("concept_set_name")
        .agg(F.count("concept_set_name").alias("count"))
        .na.fill(0)
    )

    return pivotDF


def get_diagnoses(condition_occurrence, concept_set_members):
    cohort_diagnosis = cohort_diagnosis_curated(condition_occurrence, concept_set_members)

    cohort_dx_features = cohort_dx_ct_features(cohort_diagnosis)

    return cohort_dx_features


if __name__ == "__main__":

    spark = get_spark_session()

    # Load data as spark DF
    concept_set_path = "data/raw_sample/concept_set_members.csv"
    concept_set_members = spark.read.csv(concept_set_path, header=True)

    condition_occurrence = spark.read.csv(
        "data/raw_sample/training/condition_occurrence.csv", header=True
    )

    condition_occurrence = condition_occurrence.withColumn(
        "condition_concept_id", condition_occurrence.condition_concept_id.cast("int")
    ).withColumn("person_id", condition_occurrence.person_id.cast("int"))
    index_range = spark.read.csv("data/intermediate/training/index_range.csv", header=True)

    cohort_diagnosis = cohort_diagnosis_curated(condition_occurrence, concept_set_members)

    cohort_dx_features = cohort_dx_ct_features(cohort_diagnosis)

    cohort_dx_features.show()

    # demographics.show()
