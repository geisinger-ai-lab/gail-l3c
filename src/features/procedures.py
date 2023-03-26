# Procedure Occurence
"""
@transform_pandas(
    Output(rid="ri.foundry.main.dataset.44ffaa66-95d6-4323-817d-4bbf1bf84f12"),
    procedure_occurrence_test=Input(rid="ri.foundry.main.dataset.88523aaa-75c3-4b55-a79a-ebe27e40ba4f"),
    procedure_occurrence_train=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
SELECT
person_id,
procedure_occurrence_id,
procedure_date,
procedure_datetime,
quantity,
provider_id,
visit_occurrence_id,
visit_detail_id,
modifier_source_value,
data_partner_id,
procedure_source_value,
procedure_concept_id,
procedure_type_concept_id,
modifier_concept_id,
procedure_source_concept_id,
procedure_concept_name,
procedure_type_concept_name,
modifier_concept_name,
procedure_source_concept_name
FROM procedure_occurrence_train
UNION ALL 
SELECT
person_id,
procedure_occurrence_id,
procedure_date,
procedure_datetime,
quantity,
provider_id,
visit_occurrence_id,
visit_detail_id,
modifier_source_value,
data_partner_id,
procedure_source_value,
procedure_concept_id,
procedure_type_concept_id,
modifier_concept_id,
procedure_source_concept_id,
procedure_concept_name,
procedure_type_concept_name,
modifier_concept_name,
procedure_source_concept_name
FROM procedure_occurrence_test
"""

# Procedure_all
"""
SELECT *
    FROM procedure_occurrence_merge as proce
    INNER JOIN concept_set_members as proc_concpt
        ON proce.procedure_concept_id = proc_concpt.concept_id
    WHERE proc_concpt.codeset_id  IN ('469361388', '629960249', '260036299', '838273021','415149730', '850548108', '129959605','429864170' )
    ORDER BY proce.person_id
"""

# All_procedure_with_time
# TODO this should use range_index instead of covid_index_date
"""
SELECT pmc.*, f.macrovisit_start_date, 
f.macrovisit_end_date,
f.index_start_date,
f.index_end_date,
f.pasc_code_after_four_weeks,
f.pasc_code_prior_four_weeks,
f.time_to_pasc,
case when pmc.procedure_date < f.index_start_date then 'before' 
     when pmc.procedure_date >= f.index_start_date and  pmc.procedure_date <= f.index_end_date then 'during' 
     when pmc.procedure_date > f.index_end_date then 'after' 
    end as feature_time_index
FROM Procedure_all pmc
LEFT JOIN covid_index_date f
ON  pmc.person_id = f.person_id 
ORDER BY pmc.person_id
"""


def procedure_time_pivoted(All_procedure_with_time):
    data = All_procedure_with_time
    pivoted_data = (
        data.groupBy(["person_id", "feature_time_index"])
        .pivot("concept_set_name")
        .agg(F.count("procedure_concept_name").alias("Procedure_count"))
        .na.fill(0)
    )

    renames = {
        "[ICU/MODS]IMV": "Ventilator_used",
        "Computed Tomography (CT) Scan": "Lungs_CT_scan",
        "Chest x-ray": "Chest_X_ray",
        "Lung Ultrasound": "Lung_Ultrasound",
        "Kostka - ECMO": "ECMO_performed",
        "echocardiogram_performed": "Echocardiogram_performed",
        "ECG_procedure": "ECG_performed",
        "Blood transfusion": "Blood_transfusion",
    }

    for colname, rename in renames.items():
        pivoted_data = pivoted_data.withColumnRenamed(colname, rename)

    return pivoted_data


def Procedure_timeindex_dataset_V1(procedure_time_pivoted):
    df = procedure_time_pivoted
    df1 = (
        df.groupBy("person_id")
        .pivot("feature_time_index")
        .agg(
            F.max("Ventilator_used").alias("Ventilator_used"),
            F.max("Lungs_CT_scan").alias("Lungs_CT_scan"),
            F.max("Chest_X_ray").alias("Chest_X_ray"),
            F.max("Lung_Ultrasound").alias("Lung_Ultrasound"),
            F.max("ECMO_performed").alias("ECMO_performed"),
            F.max("ECG_performed").alias("ECG_performed"),
            F.max("Echocardiogram_performed").alias("Echocardiogram_performed"),
            F.max("Blood_transfusion").alias("Blood_transfusion"),
        )
        .na.fill(0)
    )
    return df1
