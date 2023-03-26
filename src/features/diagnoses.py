# Condition_occurence
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
"""
SELECT *
    FROM condition_occurrence_merge as cond 
    INNER JOIN concept_set_members as set_cond 
        ON cond.condition_concept_id = set_cond.concept_id
    WHERE set_cond.codeset_id IN ('834391873', '882775108', '18918743', '248468138', '581513221', '628969102', '602584947','33199070')
    ORDER BY cond.person_id 

--  WHERE set_cond.concept_set_name IN ('HYPERTENSION', 'HEART FAILURE', 'DIABETES COMPLICATED', 'DIABETES UNCOMPLICATED', 'OBESITY', 'TOBACCO SMOKER', '[N3C][GAIL]asthma','[L3C][GAIL] COPD')
--  ORDER BY cond.person_id 

"""

# from pyspark.sql.functions import expr


def Cohort_dx_ct_features(COHORT_DIAGNOSIS_CURATED):
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
