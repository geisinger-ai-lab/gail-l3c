"""
Home for global functions used across the codebase 
"""

import logging
import os
import sys
from typing import Text, Union

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

## Demographics Features:
feat_demo = [
    "age",
    "FEMALE",
    "MALE",
    "Asian",
    "Asian_Indian",
    "Black_or_African_American",
    "Native_Hawaiian_or_Other_Pacific_Islander",
    "White",
]

## Measurement Features:
feat_meas = [
    "body_height_value",
    "body_weight_value",
    "alanine_aminotransferase_value",
    "albumin_value",
    "albumin_bcg_value",
    "albumin_bcp_value",
    "albumin_electrophoresis_value",
    "albumin_globulin_ratio_value",
    "alkaline_phosphatase_value",
    "anion_gap_value",
    "aspartate_aminotransferase_value",
    "bicarbonate_value",
    "bilirubin_total_value",
    "bun_value",
    "bun_creatinine_ratio_value",
    "calcium_value",
    "carbon_dioxide_total_value",
    "chloride_value",
    "creatinine_value",
    "globulin_value",
    "glomerular_filt_CKD_value",
    "glomerular_filt_blacks_CKD_value",
    "glomerular_filt_blacks_MDRD_value",
    "glomerular_filt_females_MDRD_value",
    "glomerular_filt_nonblacks_CKD_value",
    "glomerular_filt_nonblacks_MDRD_value",
    "glucose_value",
    "potassium_value",
    "protein_value",
    "sodium_value",
    "absolute_band_neutrophils_value",
    "absolute_basophils_value",
    "absolute_eosinophils_value",
    "absolute_lymph_value",
    "absolute_monocytes_value",
    "absolute_neutrophils_value",
    "absolute_other_value",
    "absolute_var_lymph_value",
    "cbc_panel_value",
    "hct_value",
    "hgb_value",
    "mch_value",
    "mchc_value",
    "mcv_value",
    "mpv_value",
    "pdw_volume_value",
    "percent_band_neutrophils_value",
    "percent_basophils_value",
    "percent_eosinophils_value",
    "percent_granulocytes_value",
    "percent_lymph_value",
    "percent_monocytes_value",
    "percent_neutrophils_value",
    "percent_other_value",
    "percent_var_lymph_value",
    "platelet_count_value",
    "rbc_count_value",
    "rdw_ratio_value",
    "rdw_volume_value",
    "wbc_count_value",
]
feat_meas_after = ["after_" + feat for feat in feat_meas]
feat_meas_before = ["before_" + feat for feat in feat_meas]
feat_meas_during = ["during_" + feat for feat in feat_meas]

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

## Diagnosis Count Features:
feat_dxct = [
    "asthma",
    "copd",
    "diabetes_complicated",
    "diabetes_uncomplicated",
    "heart_failure",
    "hypertension",
    "obesity",
]

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

## Procedure Count Features:
feat_proc = [
    "after_Ventilator_used",
    "after_Lungs_CT_scan",
    "after_Chest_X_ray",
    "after_Lung_Ultrasound",
    "after_ECMO_performed",
    "after_ECG_performed",
    "after_Echocardiogram_performed",
    "after_Blood_transfusion",
    "before_Ventilator_used",
    "before_Lungs_CT_scan",
    "before_Chest_X_ray",
    "before_Lung_Ultrasound",
    "before_ECMO_performed",
    "before_ECG_performed",
    "before_Echocardiogram_performed",
    "before_Blood_transfusion",
    "during_Ventilator_used",
    "during_Lungs_CT_scan",
    "during_Chest_X_ray",
    "during_Lung_Ultrasound",
    "during_ECMO_performed",
    "during_ECG_performed",
    "during_Echocardiogram_performed",
    "during_Blood_transfusion",
]

## Utilization features:
feat_utl = [
    "is_index_ed",
    "is_index_ip",
    "is_index_tele",
    "is_index_op",
    "avg_los",
    "avg_icu_los",
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


## Variable initialization for modeling:
random_seed = 16

## Deciding what features to run and for what label:
features = (
    feat_demo
    + feat_vitals
    + feat_meas_before
    + feat_meas_during
    + feat_meas_after
    + feat_smoke
    + feat_dxct
    + feat_meds
    + feat_utl
    + feat_proc
)

label = "pasc_code_after_four_weeks"


def add_missing_cols(df, col_list, fill_val=0):
    missing_cols = [c for c in col_list if c not in df.columns]

    select_statement = []
    for col in missing_cols:
        select_statement.append(lit(fill_val).alias(col))

    df = df.select(*df.columns, *select_statement)

    return df


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
