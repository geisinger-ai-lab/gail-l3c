"""
Functions to gather measurements and aggregate the values over three timelines with respect to the COVID-19 diagnosis (after, before, during).
"""
## Foundational libraries: 
from itertools import chain
import numpy as np 
import pandas as pd
from src.common import get_spark_session, get_index_range

## Pyspark associated libraries:
from pyspark.sql import functions as F
from pyspark.sql.functions import col, create_map, lit, when
from pyspark.sql.types import NullType

###############################################
## HEIGHT AND WEIGHT MEASUREMENTS:
###############################################
# COHORT_WT_HT_CURATED
def get_height_weight_measurements(measurement, concept_set_members):
    ## Creating a tempview:
    measurement.createOrReplaceTempView("measurement")
    concept_set_members.createOrReplaceTempView("concept_set_members")

    ## Extracting height and weight measurements using OMOP concept IDs:
    # TODO: Resolve the problem of missing measurement_concept_name: 
    expr = """
            SELECT LABS.person_id, 
                LABS.measurement_date, 
                -- LABS.measurement_concept_name, 
                -- LABS.measurement_source_concept_name,
                LOOKUP.concept_name AS measurement_concept_name,
                LABS.value_as_number, 
                -- LABS.unit_concept_name, 
                LABS.unit_concept_id,
                LABS.range_low, LABS.range_high
                FROM measurement AS LABS
                RIGHT JOIN concept_set_members AS LOOKUP
                    ON LABS.measurement_concept_id = LOOKUP.concept_id 
                WHERE LOOKUP.codeset_id IN ( 
                                            854721978, --BodyWeight
                                            186671483 --Height
                                            ) 
                    AND LABS.person_id IS NOT NULL
                    AND LABS.value_as_number IS NOT NULL
                ORDER BY LABS.person_id, LABS.measurement_date
            """
    df = spark.sql(expr)

    ## TODO: Placeholder to resolve the problem of missing unit_concept_name: 
    if 'unit_concept_name' not in df.columns:
        df = df.withColumn('unit_concept_name', lit(None).cast('string') )
    return df

def COHORT_WT_HT_PIVOT(COHORT_WT_HT_CURATED):
    """
    Standardising and Pivoting the weight and height information for the cohort by the encounter. 
    """
    ## Spark dataframe for the comprehensive metabolic panel:
    df = COHORT_WT_HT_CURATED
    ## Renaming the concept names to map to columns for pivoting the table:
    columns = {
        "Body weight": "body_weight",
        "Body weight Stated": "body_weight",
        "Body weight - Reported --usual": "body_weight",
        "Body weight Measured": "body_weight",
        "Body weight Measured --ante partum": "body_weight_msc",
        "Dry body weight Estimated": "body_weight_msc",
        "Body weight Measured --pre pregnancy": "body_weight_msc",
        "Dry body weight Measured": "dry_body_height",
        "Body weight Measured --when specimen taken": "body_weight_msc",
        "Body weight --used for drug calculation": "body_weight_msc",
        "Body height Measured": "body_height",
        "Body height Stated": "body_height",
        "Body height measure": "body_height",
        "Body height": "body_height",
        "Body height --standing": "body_height",
    }
    df2 = df.replace(to_replace=columns, subset=["measurement_concept_name"])
    ## Pivoting Table:
    pivotDF = (
        df2.groupBy(["person_id", "measurement_date"])
        .pivot("measurement_concept_name")
        .agg(
            F.first("value_as_number").alias("value"),
            F.first("unit_concept_name").alias("unit"),
        )
    )
    ## TODO: Fail safe added to ensure the minimum required columns exist:
    if 'body_height_value' not in pivotDF.columns:
        pivotDF = pivotDF.withColumn('body_height_value', lit(None).cast('string') )
    if 'body_weight_value' not in pivotDF.columns:
        pivotDF = pivotDF.withColumn('body_weight_value', lit(None).cast('string') )
    if 'body_height_unit' not in pivotDF.columns:
        pivotDF = pivotDF.withColumn('body_height_unit', lit(None).cast('string') )
    if 'body_weight_unit' not in pivotDF.columns:
        pivotDF = pivotDF.withColumn('body_weight_unit', lit(None).cast('string') )

    return pivotDF

def HT_WT_standard(COHORT_WT_HT_PIVOT):
    """
    Conversion of values to standardize the measurements associated with different units. 
    """
    ## Spark dataframe for HT WT:
    df = COHORT_WT_HT_PIVOT
    ## Standardizing height values (standard unit=centimeter):
    df3 = df.withColumn(
        "body_height_value",
        when(col("body_height_unit") == "inch (US)", col("body_height_value") * 2.54)
        .when(
            col("body_height_unit") == "inch (international)",
            col("body_height_value") * 2.54,
        )
        .when(col("body_height_unit") == "Inches", col("body_height_value") * 2.54)
        .when(col("body_height_unit") == "meter", col("body_height_value") * 100)
        .when(col("body_height_unit").isNull(), None)
        .when(col("body_height_unit") == "No matching concept", None)
        .when(col("body_height_unit") == "percent", None)
        .when(col("body_height_unit") == "milliliter", None)
        .otherwise(col("body_height_value")),
    )
    ## Standardised Height Unit (standard unit=centimeter):
    df3 = df3.withColumn(
        "body_height_unit",
        when(col("body_height_unit") == "inch (US)", "centimeter")
        .when(col("body_height_unit") == "inch (international)", "centimeter")
        .when(col("body_height_unit") == "Inches", "centimeter")
        .when(col("body_height_unit") == "meter", "centimeter")
        .when(col("body_height_unit") == "No matching concept", None)
        .when(col("body_height_unit") == "percent", None)
        .when(col("body_height_unit") == "milliliter", None)
        .otherwise(col("body_height_unit")),
    )
    ## Standardised Weight Values (standard unit=pound (US)):
    df3 = df3.withColumn(
        "body_weight_value",
        when(col("body_weight_unit") == "kilogram", col("body_weight_value") * 2.20462)
        .when(
            col("body_weight_unit") == "ounce (avoirdupois)",
            col("body_weight_value") * 0.0625,
        )
        .when(col("body_weight_unit") == "gram", col("body_weight_value") * 0.00220462)
        .when(col("body_weight_unit") == "fluid ounce (US)", None)
        .when(col("body_weight_unit") == "oz", None)
        .when(col("body_weight_unit") == "No matching concept", None)
        .when(col("body_weight_unit") == "percent", None)
        .when(col("body_weight_unit") == "meter", None)
        .when(col("body_weight_unit") == "milliliter", None)
        .when(col("body_weight_unit").isNull(), None)
        .otherwise(col("body_weight_value")),
    )
    ## Standardised Weight Unit (standard unit=pound (US)):
    df3 = df3.withColumn(
        "body_weight_unit",
        when(col("body_weight_unit") == "kilogram", "pound (US)")
        .when(col("body_weight_unit") == "ounce (avoirdupois)", "pound (US)")
        .when(col("body_weight_unit") == "gram", "pound (US)")
        .when(col("body_weight_unit") == "fluid ounce (US)", None)
        .when(col("body_weight_unit") == "oz", None)
        .when(col("body_weight_unit") == "No matching concept", None)
        .when(col("body_weight_unit") == "percent", None)
        .when(col("body_weight_unit") == "meter", None)
        .when(col("body_weight_unit") == "milliliter", None)
        .otherwise(col("body_weight_unit")),
    )
    ## Dropping unrequired columns:
    df3 = df3.drop(
        "body_weight_msc_value",
        "body_weight_msc_unit",
        "dry_body_height_unit",
        "dry_body_height_value",
    )

    return df3

###############################################
## COMPLETE BLOOD COUNT MEASUREMENTS:
###############################################
def get_cbc_measurements(measurement, concept_set_members):
    ## Creating a tempview:
    measurement.createOrReplaceTempView("measurement")
    concept_set_members.createOrReplaceTempView("concept_set_members")

    ## Extracting CBC measurements using OMOP concept IDs:
    # TODO: Resolve the problem of missing measurement_concept_name: 
    expr = """
            SELECT LABS.person_id, 
                LABS.measurement_date, 
                -- LABS.measurement_concept_name, 
                -- LABS.measurement_source_concept_name,
                LOOKUP.concept_name AS measurement_concept_name,
                LABS.value_as_number, 
                -- LABS.unit_concept_name, 
                LABS.unit_concept_id,
                LABS.range_low, LABS.range_high
                FROM measurement AS LABS
                RIGHT JOIN concept_set_members AS LOOKUP
                    ON LABS.measurement_concept_id = LOOKUP.concept_id
                WHERE LOOKUP.codeset_id IN ( 985622897 ) -- Complete Blood Panel  
                    AND LABS.person_id IS NOT NULL
                    AND LABS.value_as_number IS NOT NULL
                ORDER BY LABS.person_id, LABS.measurement_date
            """
    df = spark.sql(expr)

    ## TODO: Placeholder to resolve the problem of missing unit_concept_name: 
    if 'unit_concept_name' not in df.columns:
        df = df.withColumn('unit_concept_name', lit(None).cast('string') )
    return df

def COHORT_CBC_VALUES_PIVOT(COHORT_CBC_PANEL_CURATED):
    ## Spark dataframe for the comprehensive metabolic panel:
    df = COHORT_CBC_PANEL_CURATED

    ## Renaming the concept names to map to columns for pivoting the table:
    columns = {
        "Hemoglobin [Mass/volume] in Blood": "hgb",
        "Erythrocytes [#/volume] in Blood by Automated count": "rbc_count",
        "Leukocytes [#/volume] in Blood by Automated count": "wbc_count",
        "Platelets [#/volume] in Blood by Automated count": "platelet_count",
        "Hematocrit [Volume Fraction] of Blood by Automated count": "hct",
        "MCHC [Mass/volume] by Automated count": "mchc",
        "MCH [Entitic mass] by Automated count": "mch",
        "MCV [Entitic volume] by Automated count": "mcv",
        "Platelet mean volume [Entitic volume] in Blood by Automated count": "mpv",
        "Erythrocyte distribution width [Entitic volume] by Automated count": "rdw_volume",
        "Erythrocyte distribution width [Ratio] by Automated count": "rdw_ratio",
        "Platelet distribution width [Entitic volume] in Blood by Automated count": "pdw_volume",
        "Neutrophils [#/volume] in Blood by Automated count": "absolute_neutrophils",
        "Neutrophils/100 leukocytes in Blood by Automated count": "percent_neutrophils",
        "Lymphocytes [#/volume] in Blood by Automated count": "absolute_lymph",
        "Lymphocytes/100 leukocytes in Blood by Automated count": "percent_lymph",
        "Eosinophils [#/volume] in Blood by Automated count": "absolute_eosinophils",
        "Eosinophils/100 leukocytes in Blood by Automated count": "percent_eosinophils",
        "Basophils [#/volume] in Blood by Automated count": "absolute_basophils",
        "Basophils/100 leukocytes in Blood by Automated count": "percent_basophils",
        "Monocytes [#/volume] in Blood by Automated count": "absolute_monocytes",
        "Monocytes/100 leukocytes in Blood by Automated count": "percent_monocytes",
        "Granulocytes/100 leukocytes in Blood by Automated count": "percent_granulocytes",
        "Other cells [#/volume] in Blood by Automated count": "absolute_other",
        "Other cells/100 leukocytes in Blood by Automated count": "percent_other",
        "Variant lymphocytes [#/volume] in Blood by Automated count": "absolute_var_lymph",
        "Variant lymphocytes/100 leukocytes in Blood by Automated count": "percent_var_lymph",
        "Band form neutrophils/100 leukocytes in Blood by Automated count": "absolute_band_neutrophils",
        "Band form neutrophils [#/volume] in Blood by Automated count": "percent_band_neutrophils",
        "CBC panel - Blood by Automated count": "cbc_panel",
    }

    df2 = df.replace(to_replace=columns, subset=["measurement_concept_name"])

    ## Pivoting Table:
    pivotDF = (
        df2.groupBy(["person_id", "measurement_date"])
        .pivot("measurement_concept_name")
        .agg(
            F.first("value_as_number").alias("value"),
            F.first("unit_concept_name").alias("unit"),
        )
    )

    for feature in list( set(val for val in columns.values()) ):
        ## Add Feature value & unit column to the DF if doesn't exist:
        pivotDF = pivotDF.withColumn(f'{feature}_value', lit(None).cast('string') )\
                    .withColumn(f'{feature}_unit', lit(None).cast('string') )

    ## Standardising Units:
    pivotDF = pivotDF.na.fill("gram per deciliter", subset=["hgb_unit", "mchc_unit"])
    pivotDF = pivotDF.replace(
        "No matching concept", "gram per deciliter", subset=["hgb_unit", "mchc_unit"]
    )
    pivotDF = pivotDF.replace("percent", "gram per deciliter", subset=["mchc_unit"])

    # remove units other than grams per deciliter
    pivotDF = pivotDF.withColumn(
        "hgb_value",
        when(col("hgb_unit") != "gram per deciliter", None).otherwise(col("hgb_value")),
    )
    pivotDF = pivotDF.withColumn(
        "mchc_value",
        when(col("mchc_unit") != "gram per deciliter", None).otherwise(
            col("mchc_value")
        ),
    )

    pivotDF = pivotDF.na.fill(
        "femtoliter", subset=["mcv_unit", "mpv_unit", "rdw_volume_unit"]
    )
    pivotDF = pivotDF.replace(
        "No matching concept",
        "femtoliter",
        subset=["mcv_unit", "mpv_unit", "rdw_volume_unit"],
    )
    # remove units other than femoliter
    pivotDF = pivotDF.withColumn(
        "mcv_value",
        when(col("mcv_unit") != "femtoliter", None).otherwise(col("mcv_value")),
    )
    pivotDF = pivotDF.withColumn(
        "mpv_value",
        when(col("mpv_unit") != "femtoliter", None).otherwise(col("mpv_value")),
    )
    pivotDF = pivotDF.withColumn(
        "rdw_volume_value",
        when(col("rdw_volume_unit") != "femtoliter", None).otherwise(
            col("rdw_volume_value")
        ),
    )

    pivotDF = pivotDF.na.fill("million per microliter", subset=["rbc_count_unit"])
    pivotDF = pivotDF.replace(
        "No matching concept", "million per microliter", subset=["rbc_count_unit"]
    )
    pivotDF = pivotDF.replace(
        "trillion per liter", "million per microliter", subset=["rbc_count_unit"]
    )
    # remove units other than million per microliter
    pivotDF = pivotDF.withColumn(
        "rbc_count_value",
        when(col("rbc_count_unit") != "million per microliter", None).otherwise(
            col("rbc_count_value")
        ),
    )

    pivotDF = pivotDF.na.fill("picogram", subset=["mch_unit"])
    pivotDF = pivotDF.replace("No matching concept", "picogram", subset=["mch_unit"])
    # remove units other than million per picogram
    pivotDF = pivotDF.withColumn(
        "mch_value",
        when(col("mch_unit") != "picogram", None).otherwise(col("mch_value")),
    )

    pivotDF = pivotDF.na.fill(
        "thousand per microliter",
        subset=[
            "absolute_basophils_unit",
            "platelet_count_unit",
            "absolute_neutrophils_unit",
            "absolute_eosinophils_unit",
            "wbc_count_unit",
            "absolute_monocytes_unit",
            "absolute_lymph_unit",
        ],
    )
    pivotDF = pivotDF.replace(
        "No matching concept",
        "thousand per microliter",
        subset=[
            "absolute_basophils_unit",
            "platelet_count_unit",
            "absolute_neutrophils_unit",
            "absolute_eosinophils_unit",
            "wbc_count_unit",
            "absolute_monocytes_unit",
            "absolute_lymph_unit",
        ],
    )
    pivotDF = pivotDF.replace(
        "billion per liter",
        "thousand per microliter",
        subset=[
            "absolute_basophils_unit",
            "platelet_count_unit",
            "absolute_neutrophils_unit",
            "absolute_eosinophils_unit",
            "wbc_count_unit",
            "absolute_monocytes_unit",
            "absolute_lymph_unit",
        ],
    )
    # remove units other than thousand per microliter
    pivotDF = pivotDF.withColumn(
        "absolute_basophils_value",
        when(
            col("absolute_basophils_unit") != "thousand per microliter", None
        ).otherwise(col("absolute_basophils_value")),
    )
    pivotDF = pivotDF.withColumn(
        "platelet_count_value",
        when(col("platelet_count_unit") != "thousand per microliter", None).otherwise(
            col("platelet_count_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "absolute_neutrophils_value",
        when(
            col("absolute_neutrophils_unit") != "thousand per microliter", None
        ).otherwise(col("absolute_neutrophils_value")),
    )
    pivotDF = pivotDF.withColumn(
        "absolute_eosinophils_value",
        when(
            col("absolute_eosinophils_unit") != "thousand per microliter", None
        ).otherwise(col("absolute_eosinophils_value")),
    )
    pivotDF = pivotDF.withColumn(
        "wbc_count_value",
        when(col("wbc_count_unit") != "thousand per microliter", None).otherwise(
            col("wbc_count_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "absolute_monocytes_value",
        when(
            col("absolute_monocytes_unit") != "thousand per microliter", None
        ).otherwise(col("absolute_monocytes_value")),
    )
    pivotDF = pivotDF.withColumn(
        "absolute_lymph_value",
        when(col("absolute_lymph_unit") != "thousand per microliter", None).otherwise(
            col("absolute_lymph_value")
        ),
    )

    pivotDF = pivotDF.na.fill(
        "percent",
        subset=[
            "rdw_ratio_unit",
            "hct_unit",
            "percent_neutrophils_unit",
            "percent_lymph_unit",
            "percent_basophils_unit",
            "percent_eosinophils_unit",
            "percent_monocytes_unit",
        ],
    )
    pivotDF = pivotDF.replace(
        "No matching concept",
        "percent",
        subset=[
            "rdw_ratio_unit",
            "hct_unit",
            "percent_neutrophils_unit",
            "percent_lymph_unit",
            "percent_basophils_unit",
            "percent_eosinophils_unit",
            "percent_monocytes_unit",
        ],
    )
    # remove units other than percent
    pivotDF = pivotDF.withColumn(
        "rdw_ratio_value",
        when(col("rdw_ratio_unit") != "percent", None).otherwise(
            col("rdw_ratio_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "hct_value",
        when(col("hct_unit") != "percent", None).otherwise(col("hct_value")),
    )
    pivotDF = pivotDF.withColumn(
        "percent_neutrophils_value",
        when(col("percent_neutrophils_unit") != "percent", None).otherwise(
            col("percent_neutrophils_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "percent_lymph_value",
        when(col("percent_lymph_unit") != "percent", None).otherwise(
            col("percent_lymph_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "percent_basophils_value",
        when(col("percent_basophils_unit") != "percent", None).otherwise(
            col("percent_basophils_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "percent_eosinophils_value",
        when(col("percent_eosinophils_unit") != "percent", None).otherwise(
            col("percent_eosinophils_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "percent_monocytes_value",
        when(col("percent_monocytes_unit") != "percent", None).otherwise(
            col("percent_monocytes_value")
        ),
    )

    ## Identify the missing features in the df.columns:
    return pivotDF

###############################################
## COMPREHENSIVE METABOLIC PROFILE MEASUREMENTS:
###############################################
def get_cmp_measurements(measurement, concept_set_members):
    ## Creating a tempview:
    measurement.createOrReplaceTempView("measurement")
    concept_set_members.createOrReplaceTempView("concept_set_members")
    ## Extracting CBC measurements using OMOP concept IDs:
    # TODO: Resolve the problem of missing measurement_concept_name: 
    expr = """
            SELECT LABS.person_id, 
                LABS.measurement_date, 
                -- LABS.measurement_concept_name, 
                -- LABS.measurement_source_concept_name,
                LOOKUP.concept_name AS measurement_concept_name,
                LABS.value_as_number, 
                -- LABS.unit_concept_name, 
                LABS.unit_concept_id,
                LABS.range_low, LABS.range_high
                FROM measurement AS LABS
                RIGHT JOIN concept_set_members AS LOOKUP
                    ON LABS.measurement_concept_id = LOOKUP.concept_id
                WHERE LOOKUP.codeset_id IN ( 212998332, --Comprehensive metabolic Profile
                                            104464584 --Albumin
                                            ) 
                    AND LABS.person_id IS NOT NULL
                    AND LABS.value_as_number IS NOT NULL
                ORDER BY LABS.person_id, LABS.measurement_date
            """
    df = spark.sql(expr)

    ## TODO: Placeholder to resolve the problem of missing unit_concept_name: 
    if 'unit_concept_name' not in df.columns:
        df = df.withColumn('unit_concept_name', lit(None).cast('string') )
    return df

def COHORT_CMP_VALUES_PIVOT(COHORT_CMP_CURATED):
    ## Spark dataframe for the comprehensive metabolic panel:
    df = COHORT_CMP_CURATED

    ## Renaming the concept names to map to columns for pivoting the table (Standardising):
    columns = {
        "Calcium [Mass/volume] in Serum or Plasma": "calcium",
        "Potassium [Moles/volume] in Serum or Plasma": "potassium",
        "Bicarbonate [Moles/volume] in Serum or Plasma": "bicarbonate",
        "Chloride [Moles/volume] in Serum or Plasma": "chloride",
        "Sodium [Moles/volume] in Serum or Plasma": "sodium",
        "Carbon dioxide, total [Moles/volume] in Serum or Plasma": "carbon_dioxide_total",
        "Protein [Mass/volume] in Serum or Plasma": "protein",
        "Glucose [Mass/volume] in Serum or Plasma": "glucose",
        "Creatinine [Mass/volume] in Serum or Plasma": "creatinine",
        "Bilirubin.total [Mass/volume] in Serum or Plasma": "bilirubin_total",
        "Globulin [Mass/volume] in Serum by calculation": "globulin",
        "Albumin/Globulin [Mass Ratio] in Serum or Plasma": "albumin_globulin_ratio",
        "Albumin [Mass/volume] in Serum or Plasma by Bromocresol green (BCG) dye binding method": "albumin_bcg",
        "Albumin [Mass/volume] in Serum or Plasma": "albumin",
        "Albumin [Mass/volume] in Serum or Plasma by Electrophoresis": "albumin_electrophoresis",
        "Albumin [Mass/volume] in Serum or Plasma by Bromocresol purple (BCP) dye binding method": "albumin_bcp",
        "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma": "alkaline_phosphatase",
        "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma": "alanine_aminotransferase",
        "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma": "aspartate_aminotransferase",
        "Anion gap in Serum or Plasma": "anion_gap",
        "Urea nitrogen [Mass/volume] in Serum or Plasma": "bun",
        "Urea nitrogen/Creatinine [Mass Ratio] in Serum or Plasma": "bun_creatinine_ratio",
        "Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)": "glomerular_filt_blacks_MDRD",
        "Glomerular filtration rate/1.73 sq M.predicted among non-blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)": "glomerular_filt_nonblacks_MDRD",
        "Glomerular filtration rate/1.73 sq M.predicted among females [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)": "glomerular_filt_females_MDRD",
        "Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)": "glomerular_filt_blacks_CKD",
        "Glomerular filtration rate/1.73 sq M.predicted among non-blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)": "glomerular_filt_nonblacks_CKD",
        "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)": "glomerular_filt_CKD",
    }
    df2 = df.replace(to_replace=columns, subset=["measurement_concept_name"])

    ## Pivoting Table:
    pivotDF = (
        df2.groupBy(["person_id", "measurement_date"])
        .pivot("measurement_concept_name")
        .agg(
            F.first("value_as_number").alias("value"),
            F.first("unit_concept_name").alias("unit"),
        )
    )

    for feature in list( set(val for val in columns.values()) ):
        ## Add Feature value & unit column to the DF if doesn't exist:
        pivotDF = pivotDF.withColumn(f'{feature}_value', lit(None).cast('string') )\
                    .withColumn(f'{feature}_unit', lit(None).cast('string') )

    ## Standardising Units:
    pivotDF = pivotDF.na.fill(
        "millimole per liter",
        subset=[
            "sodium_unit",
            "chloride_unit",
            "potassium_unit",
            "carbon_dioxide_total_unit",
        ],
    )
    pivotDF = pivotDF.replace(
        "No matching concept",
        "millimole per liter",
        subset=[
            "sodium_unit",
            "chloride_unit",
            "potassium_unit",
            "carbon_dioxide_total_unit",
        ],
    )
    # remove units other than millimole per liter
    pivotDF = pivotDF.withColumn(
        "sodium_value",
        when(col("sodium_unit") != "millimole per liter", None).otherwise(
            col("sodium_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "chloride_value",
        when(col("chloride_unit") != "millimole per liter", None).otherwise(
            col("chloride_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "potassium_value",
        when(col("potassium_unit") != "millimole per liter", None).otherwise(
            col("potassium_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "carbon_dioxide_total_value",
        when(col("carbon_dioxide_total_unit") != "millimole per liter", None).otherwise(
            col("carbon_dioxide_total_value")
        ),
    )

    pivotDF = pivotDF.na.fill(
        "milligram per deciliter",
        subset=[
            "bun_unit",
            "calcium_unit",
            "creatinine_unit",
            "glucose_unit",
            "bilirubin_total_unit",
        ],
    )
    pivotDF = pivotDF.replace(
        "No matching concept",
        "milligram per deciliter",
        subset=[
            "bun_unit",
            "calcium_unit",
            "creatinine_unit",
            "glucose_unit",
            "bilirubin_total_unit",
        ],
    )
    # remove units other than milligram per deciliter
    pivotDF = pivotDF.withColumn(
        "bun_value",
        when(col("bun_unit") != "milligram per deciliter", None).otherwise(
            col("bun_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "calcium_value",
        when(col("calcium_unit") != "milligram per deciliter", None).otherwise(
            col("calcium_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "creatinine_value",
        when(col("creatinine_unit") != "milligram per deciliter", None).otherwise(
            col("creatinine_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "glucose_value",
        when(col("glucose_unit") != "milligram per deciliter", None).otherwise(
            col("glucose_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "bilirubin_total_value",
        when(col("bilirubin_total_unit") != "milligram per deciliter", None).otherwise(
            col("bilirubin_total_value")
        ),
    )

    pivotDF = pivotDF.na.fill(
        "unit per liter",
        subset=[
            "alkaline_phosphatase_unit",
            "aspartate_aminotransferase_unit",
            "alanine_aminotransferase_unit",
        ],
    )
    pivotDF = pivotDF.replace(
        "international unit per liter",
        "unit per liter",
        subset=[
            "alkaline_phosphatase_unit",
            "aspartate_aminotransferase_unit",
            "alanine_aminotransferase_unit",
        ],
    )
    pivotDF = pivotDF.replace(
        "No matching concept",
        "unit per liter",
        subset=["aspartate_aminotransferase_unit", "alanine_aminotransferase_unit"],
    )
    # remove units other than unit per liter
    pivotDF = pivotDF.withColumn(
        "alkaline_phosphatase_value",
        when(col("alkaline_phosphatase_unit") != "unit per liter", None).otherwise(
            col("alkaline_phosphatase_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "aspartate_aminotransferase_value",
        when(
            col("aspartate_aminotransferase_unit") != "unit per liter", None
        ).otherwise(col("aspartate_aminotransferase_value")),
    )
    pivotDF = pivotDF.withColumn(
        "alanine_aminotransferase_value",
        when(col("alanine_aminotransferase_unit") != "unit per liter", None).otherwise(
            col("alanine_aminotransferase_value")
        ),
    )

    pivotDF = pivotDF.na.fill(
        "gram per deciliter", subset=["protein_unit", "albumin_unit"]
    )
    pivotDF = pivotDF.replace(
        "No matching concept",
        "gram per deciliter",
        subset=["protein_unit", "albumin_unit"],
    )
    # remove units other than milligram per deciliter
    pivotDF = pivotDF.withColumn(
        "protein_value",
        when(col("protein_unit") != "gram per deciliter", None).otherwise(
            col("protein_value")
        ),
    )
    pivotDF = pivotDF.withColumn(
        "albumin_value",
        when(col("albumin_unit") != "gram per deciliter", None).otherwise(
            col("albumin_value")
        ),
    )

    return pivotDF

###############################################
## IMMUNOASSAY MEASUREMENTS:
###############################################
"""
# COHORT_IMMUNOASSAY_CURATED
    """
    #SELECT LABS.person_id, 
    #LABS.measurement_date, LABS.measurement_concept_name, LABS.measurement_source_concept_name,
    #LABS.value_as_number, LABS.unit_concept_name, LABS.range_low, LABS.range_high
    #FROM measurement_merge AS LABS
    #RIGHT JOIN concept_set_members AS LOOKUP
    #    ON LABS.measurement_concept_id = LOOKUP.concept_id
    #WHERE LOOKUP.codeset_id IN ( 459475527 ) -- ImmunoAssay 
    #    AND LABS.person_id IS NOT NULL
    #    AND LABS.value_as_number IS NOT NULL
    #ORDER BY LABS.person_id, LABS.measurement_date
"""

def COHORT_IMMUNOGLOBIN_PIVOT(COHORT_IMMUNOASSAY_CURATED):
    ## Spark dataframe for the comprehensive metabolic panel:
    df = COHORT_IMMUNOASSAY_CURATED

    ## Renaming the concept names to map to columns for pivoting the table:
    columns = {
        "SARS-CoV-2 (COVID-19) Ab [Presence] in Serum or Plasma by Immunoassay": "cov2_Ab_presence",
        "SARS-CoV-2 (COVID-19) Ab panel - Serum or Plasma by Immunoassay": "cov2_Ab_panel",
        "SARS-CoV-2 (COVID-19) Ab [Units/volume] in Serum or Plasma by Immunoassay": "cov2_Ab_amt",
        "SARS-CoV-2 (COVID-19) Ag [Presence] in Respiratory specimen by Rapid immunoassay": "cov2_Ag_presence_rapid",
        "SARS-CoV-2 (COVID-19) Ag [Presence] in Upper respiratory specimen by Immunoassay": "cov2_Ag_presence",
        "SARS-CoV-2 (COVID-19) Ag [Presence] in Upper respiratory specimen by Rapid immunoassay": "cov2_Ag_presence_rapid",
        "SARS-CoV+SARS-CoV-2 (COVID-19) Ag [Presence] in Respiratory specimen by Rapid immunoassay": "cov2_Ag_presence",
        "SARS-CoV-2 (COVID-19) IgG Ab [Presence] in Serum, Plasma or Blood by Rapid immunoassay": "cov2_IgG_Ab_presence_rapid",
        "SARS-CoV-2 (COVID-19) IgG Ab [Presence] in DBS by Immunoassay": "cov2_IgG_Ab_presence_DBS",
        "SARS-CoV-2 (COVID-19) IgG Ab [Presence] in Serum or Plasma by Immunoassay": "cov2_IgG_Ab_presence",
        "SARS-CoV-2 (COVID-19) IgG Ab [Units/volume] in Serum or Plasma by Immunoassay": "cov2_IgG_Ab_amt",
        "SARS-CoV-2 (COVID-19) IgA Ab [Presence] in Serum or Plasma by Immunoassay": "cov2_IgA_Ab_presence",
        "Influenza virus A and B and SARS-CoV+SARS-CoV-2 (COVID-19) Ag panel - Upper respiratory specimen by Rapid immunoassay": "influ_cov2_Ag_panel",
        "Influenza virus A Ag [Presence] in Upper respiratory specimen by Rapid immunoassay": "influA_Ag_presence",
        "Influenza virus B Ag [Presence] in Upper respiratory specimen by Rapid immunoassay": "influB_Ag_presence",
        "SARS-CoV-2 (COVID-19) IgM Ab [Units/volume] in Serum or Plasma by Immunoassay": "cov2_IgM_Ab_amt",
        "SARS-CoV-2 (COVID-19) IgM Ab [Presence] in Serum, Plasma or Blood by Rapid immunoassay": "cov2_IgA_Ab_presence_rapid",
        "SARS-CoV-2 (COVID-19) IgM Ab [Presence] in Serum or Plasma by Immunoassay": "cov2_IgA_Ab_presence",
        "SARS-CoV-2 (COVID-19) IgG+IgM Ab [Presence] in Serum or Plasma by Immunoassay": "cov2_IgG_IgM_Ab_presence",
    }

    df2 = df.replace(to_replace=columns, subset=["measurement_concept_name"])

    ## Pivoting Table:
    pivotDF = (
        df2.groupBy(["person_id", "measurement_date"])
        .pivot("measurement_concept_name")
        .agg(F.first("value_as_number").alias("value"))
        .orderBy(["person_id", "measurement_date"])
    )

    return pivotDF
"""

def measurements_standard_features(HT_WT_standard, COHORT_CBC_VALUES_PIVOT, COHORT_CMP_VALUES_PIVOT, index_range):
    from re import match

    ## Initialize all measurement related features:
    df_htwt = HT_WT_standard.withColumnRenamed(
        "person_id", "person_id_htwt"
    ).withColumnRenamed("measurement_date", "measurement_date_htwt")
    df_cbc = COHORT_CBC_VALUES_PIVOT.withColumnRenamed(
        "person_id", "person_id_cbc"
    ).withColumnRenamed("measurement_date", "measurement_date_cbc")
    df_cmp = COHORT_CMP_VALUES_PIVOT.withColumnRenamed(
        "person_id", "person_id_cmp"
    ).withColumnRenamed("measurement_date", "measurement_date_cmp")
    df_covid = index_range.withColumnRenamed("person_id", "person_id_covid")

    ## Joins:
    df_feat = df_htwt.join(
        df_cmp,
        [
            df_htwt.person_id_htwt == df_cmp.person_id_cmp,
            df_htwt.measurement_date_htwt == df_cmp.measurement_date_cmp,
        ],
        "outer",
    ).join(
        df_cbc,
        [
            df_htwt.person_id_htwt == df_cbc.person_id_cbc,
            df_htwt.measurement_date_htwt == df_cbc.measurement_date_cbc,
        ],
        "outer",
    )

    ## generating a common person_id and measurement_date column
    df_feat = df_feat.withColumn(
        "person_id",
        F.greatest(
            F.col("person_id_htwt"), F.col("person_id_cmp"), F.col("person_id_cbc")
        ),
    )
    df_feat = df_feat.withColumn(
        "measurement_date",
        F.greatest(
            F.col("measurement_date_htwt"),
            F.col("measurement_date_cmp"),
            F.col("measurement_date_cbc"),
        ),
    )

    ## Adding a new column for before, during, after:
    df_feat = df_feat.join(
        df_covid.select("person_id_covid", "index_start_date", "index_end_date"),
        df_feat.person_id == df_covid.person_id_covid,
        "left",
    )
    df_feat = df_feat.withColumn(
        "status_relative_index",
        when(col("measurement_date") > col("index_end_date"), "after")
        .when(col("measurement_date") < col("index_start_date"), "before")
        .otherwise("during"),
    )

    ## Dropping unnecessary columns:
    cols = df_feat.columns
    filtered_cols = list(filter(lambda v: match(".*unit$", v), cols))
    df_feat = df_feat.drop(
        "person_id_htwt",
        "measurement_date_htwt",
        "person_id_cbc",
        "measurement_date_cbc",
        "person_id_cmp",
        "measurement_date_cmp",
        "person_id_covid",
        "index_start_date",
        "index_end_date",
        "measurement_date",
        *filtered_cols
    )

    return df_feat

def measurements_standard_features_final(measurements_standard_features):
    df = measurements_standard_features
    features = df.columns
    features.remove("person_id")
    features.remove("status_relative_index")

    ## Pivoting:
    df_grouped = df.groupBy("person_id", "status_relative_index").mean()

    for name in features:
        df_grouped = df_grouped.withColumnRenamed("avg(" + name + ")", name)
    
    for feature in features:
        ## Add Feature value & unit column to the DF if doesn't exist:
        df_grouped = df_grouped.withColumn( f'{feature}', lit(None).cast('string') )

    return df_grouped

def Measurements_after(measurements_standard_features_final):
    df = measurements_standard_features_final
    features = df.columns
    features.remove("status_relative_index")

    return df.filter(df.status_relative_index == "after").select(*features)

def Measurements_during(measurements_standard_features_final):
    df = measurements_standard_features_final
    features = df.columns
    features.remove("status_relative_index")

    return df.filter(df.status_relative_index == "during").select(*features)

def Measurements_before(measurements_standard_features_final):
    df = measurements_standard_features_final
    features = df.columns
    features.remove("status_relative_index")

    return df.filter(df.status_relative_index == "before").select(*features)

## MAIN SCRIPT:
if __name__ == "__main__":

    ## Start a Spark Session:
    spark = get_spark_session()

    ## File path and directory pathe initializations:
    lab_set_path = "data/raw_sample/training/measurement.csv"
    concept_set_path = "data/raw_sample/concept_set_members.csv"

    ## Load measurements data into a spark dataframe:
    measurement = spark.read.csv(lab_set_path, header=True)
    measurement = measurement.withColumn("measurement_concept_id", measurement.measurement_concept_id.cast("int"))\
                    .withColumn("unit_concept_id", measurement.unit_concept_id.cast("int"))
    #print(f"Column Names: {measurement.columns}")
    #measurement.show(5)
    
    ## Loading the concept members to filter the desired measurements:
    concept_set_members = spark.read.csv(concept_set_path, header=True)
    concept_set_members = concept_set_members.withColumn("concept_id", concept_set_members.concept_id.cast("int"))
    #concept_set_members.show(5)

    ## Height and weight measurements:  
    # Step 1: Filtering the height and weight measurements:
    df_ht_wt_curated = get_height_weight_measurements(measurement, concept_set_members)
    #df_ht_wt_curated.show(5)
    # Step 2: Pivoting the dataframe and standardising units:
    df_ht_wt_pivot = COHORT_WT_HT_PIVOT(df_ht_wt_curated)
    df_ht_wt_standard = HT_WT_standard(df_ht_wt_pivot)
    #df_ht_wt_standard.show(5)

    ## CBC Panel Measurements:
    # Step 1: Filtering the height and weight measurements:
    df_cbc_curated = get_cbc_measurements(measurement, concept_set_members)
    # Step 2: Pivoting the dataframe and standardising units:
    df_cbc_pivot = COHORT_CBC_VALUES_PIVOT(df_cbc_curated)
    #df_cbc_pivot.show()

    ## CBC Panel Measurements:
    # Step 1: Filtering the height and weight measurements:
    df_cmp_curated = get_cmp_measurements(measurement, concept_set_members)
    # Step 2: Pivoting the dataframe and standardising units:
    df_cmp_pivot = COHORT_CMP_VALUES_PIVOT(df_cmp_curated)
    #df_cmp_pivot.show()

    ## Index Range and featurizing:
    index_range = spark.read.csv(
        "data/intermediate/training/index_range.csv", header=True
    )
    df_measurement_features = measurements_standard_features(df_ht_wt_standard, df_cbc_pivot, df_cmp_pivot, index_range)
    #df_measurement_features.show()
    df_measurement_features_final = measurements_standard_features_final(df_measurement_features)
    df_measurement_features_after = Measurements_after(df_measurement_features_final)
    df_measurement_features_during = Measurements_during(df_measurement_features_final)
    df_measurement_features_before = Measurements_before(df_measurement_features_final)

    #df_measurement_features_final.show()
    df_measurement_features_after.show()
    df_measurement_features_during.show() 
    df_measurement_features_before.show()