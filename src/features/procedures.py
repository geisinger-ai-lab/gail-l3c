"""
function to load data (concepts, drug_exposure, index_range(derrived))

Implement the sql steps as functions that take in dataframes

"public" function that returns the formatted med features, 
    with path parameters (e.g. train/test)

"""

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from src.common import get_spark_session, rename_cols


def get_procedure_all(concept_set_members,procedure_occurrence):
    """
    runs SQL on concept set members

    Expects concept_set_members to be a spark dataframe

    Returns a spark dataframe of procedure-related data
    """
    concept_set_members.createOrReplaceTempView("concept_set_members")
    procedure_occurrence.createOrReplaceTempView("procedure_occurrence")

    procedure_sql = """SELECT *
    FROM procedure_occurrence as proce
    INNER JOIN concept_set_members as proc_concpt
        ON proce.procedure_concept_id = proc_concpt.concept_id
    WHERE proc_concpt.codeset_id  IN ('469361388', '629960249', '260036299', '838273021','415149730', '850548108', '129959605','429864170' )
    ORDER BY proce.person_id"""
    return spark.sql(procedure_sql)


def all_procedure_with_time(index_range, procedure_all):
    """
    Runs Spark-flavor SQL code to connect all the procedures with the relevent time index
    """
    index_range.createOrReplaceTempView("index_range")
    procedure_all.createOrReplaceTempView("procedure_all")
   
    sql = """SELECT pmc.*, f.macrovisit_start_date, 
f.macrovisit_end_date,
f.index_start_date,
f.index_end_date,
f.pasc_code_after_four_weeks,
f.time_to_pasc,
case when pmc.procedure_date < f.index_start_date then 'before' 
     when pmc.procedure_date >= f.index_start_date and  pmc.procedure_date <= f.index_end_date then 'during' 
     when pmc.procedure_date > f.index_end_date then 'after' 
    end as feature_time_index
FROM procedure_all pmc
LEFT JOIN index_range f
ON  pmc.person_id = f.person_id 
ORDER BY pmc.person_id
"""
    return spark.sql(sql)


def procedure_time_grouped(all_procedure_with_time):

    """
    Procedure grouped, needs to be a PySpark function
    
    """
     
    from pyspark.sql.functions import lit
    data = all_procedure_with_time
    
    pivoted_data = data.groupBy(["person_id","feature_time_index"]).pivot("concept_set_name").agg(F.count("procedure_concept_id").alias("Procedure_count")).na.fill(0)
    
    renames = {
    "[ICU/MODS]IMV": "Ventilator_used",
    "Computed Tomography (CT) Scan": "Lungs_CT_scan",
    "Chest x-ray": "Chest_X_ray",
    "Lung Ultrasound": "Lung_Ultrasound",
    "Kostka - ECMO": "ECMO_performed",
    "echocardiogram_performed": "Echocardiogram_performed",
    "ECG_procedure": "ECG_performed",
    "Blood transfusion": "Blood_transfusion"    
}

    for colname, rename in renames.items():
        pivoted_data  = pivoted_data .withColumnRenamed(colname, rename)

    all_columns = ["person_id","feature_time_index","Blood_transfusion","Chest_X_ray","Lungs_CT_scan","ECG_performed","ECMO_performed","Lung_Ultrasound","Ventilator_used","Echocardiogram_performed"]
    proc_data_feature = pivoted_data.columns
    missingFeature = list(set(all_columns )- set(proc_data_feature))
    
    for i in missingFeature:
          pivoted_data = pivoted_data .withColumn(i, lit(0))

    return pivoted_data
                                    
    
def proc_dataset(proc_grouped):
    
    df =  proc_grouped
    df1 = df.groupBy("person_id"). pivot("feature_time_index"). agg(F.max("Ventilator_used").alias('Ventilator_used'),
F.max('Lungs_CT_scan').alias('Lungs_CT_scan'),F.max('Chest_X_ray').alias('Chest_X_ray'),F.max('Lung_Ultrasound').alias('Lung_Ultrasound'),
F.max('ECG_performed').alias('ECG_performed'), F.max('ECMO_performed').alias('ECMO_performed'),
F.max('Blood_transfusion').alias('Blood_transfusion'), F.max('Echocardiogram_performed').alias
('Echocardiogram_performed')).na.fill(0)
    return df1


def get_procedure_dataset(concept_set_members, procedure_occurrence, index_range):
    """
    The "Public" function, meant to be called from the main data preparation script

    Returns formatted procedure features
    """

    procedure_all = get_procedure_all(concept_set_members,procedure_occurrence)
    procedure_with_time = all_procedure_with_time(index_range, procedure_all)
    procedure_grouped = procedure_time_grouped(procedure_with_time)
    procedure_df = proc_dataset(procedure_grouped)

    return procedure_df


if __name__ == "__main__":

    spark = get_spark_session()

    # Load data as spark DF
    concept_set_path = "data/raw_sample/concept_set_members.csv"
    concept_set_members = spark.read.csv(concept_set_path, header=True)

    procedure_occurrence = spark.read.csv(
        "data/raw_sample/training/procedure_occurrence.csv", header=True, inferSchema=True
    )
    index_range = spark.read.csv(
        "data/intermediate/training/index_range.csv", header=True, inferSchema=True
    )

    # Run the procedure data ETL
    procedure_df = get_procedure_dataset(concept_set_members, procedure_occurrence, index_range)

    procedure_df.show()