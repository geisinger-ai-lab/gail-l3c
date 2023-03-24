"""
function to load data (concepts, drug_exposure, index_range(derrived))

Implement the sql steps as functions that take in dataframes

"public" function that returns the formatted med features, 
    with path parameters (e.g. train/test)

"""

import pandas as pd

from pyspark.sql import functions as F

from somewhere_dbd import rename_cols

def get_concept_set_members():
    pass

def get_meds_concepts(concept_set_members):
    """
    runs SQL on concept set members
    """
    meds_sql = """select *
    , 'anticoagulants' as feature_name
    , 'before_after' as temporality
from concept_set_members csm
where csm.codeset_id = 761556952 -- [DM] Anticoagulants
union
 select *
    , 'asthma_drugs' as feature_name
    , 'before_after' as temporality
from concept_set_members csm
where csm.codeset_id = 56408680 -- Asthma_drugs_wide
union
select *
    , 'corticosteroids' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 804872085 -- NIH Systemic Corticosteroids
union
select *
    , 'antivirals' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 245992624 -- Systemic Antiviral Medications
union
select *
    , 'lopinavir' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 869338732 -- N3C lopinavir / ritonavir 
union
select *
    , 'antibiotics' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 54810628 -- Systemic Antibiotics
union
select *
    , 'thymosin' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 358954641 -- [L3C][GAIL] Thymosin (v1)
union
select *
    , 'umifenovir' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 33014037 -- [L3C][GAIL] Umifenovir (v1)
union
select *
    , 'iv_immunoglobulin' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 241604784 -- Intravenous Immunoglobulin (IVIG) (v1)
union
select *
    , 'remdesivir' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 719693192 -- Remdesivir (v2)
union
select *
    , 'paxlovid' as feature_name
    , 'during' as temporality
from concept_set_members csm
where csm.codeset_id = 339287218 -- paxlovid (v1)"""
    pass

def get_drug_exposure(drug_exposure_path):
    """
    Get either the train or test drug exposure table
    """
    pass

def get_index_range(index_range_path):
    pass

def filter_meds(index_range, meds_concepts, drug_exposure):
    sql = """select
    d.person_id
    , d.drug_concept_id
    , d.drug_exposure_start_date
    , d.drug_type_concept_id
    , d.drug_type_concept_name
    , mc.feature_name
    , explode(split(mc.temporality, '_')) as temporality
    , case 
        when (d.drug_exposure_start_date >= ci.index_start_date) 
            and (d.drug_exposure_start_date <= ci.index_end_date)   then 'during'
        when d.drug_exposure_start_date < ci.index_start_date       then 'before'
        when d.drug_exposure_start_date > ci.index_end_date         then 'after'
    end as before_or_after_index
from        drug_exposure_merge   as d
left join   meds_concepts   as mc on d.drug_concept_id = mc.concept_id
left join   index_range     as ci on ci.person_id = d.person_id
where mc.feature_name is not null
"""
    pass

def group_meds():
    sql = """select 
    m.person_id
    , m.before_or_after_index
    , m.feature_name
    , count(m.person_id) as med_count
from 
    meds_filtered m
where 
    m.temporality = m.before_or_after_index
group by 
    m.person_id
    , m.before_or_after_index
    , m.feature_name"""
    pass 

def Meds_dataset_v1(meds_grouped):
    """
    Meds grouped needs to be a PySpark function
    also need to have imported rename_cols from global somewhere 
    """
    # Split by templorality
    meds_before = meds_grouped.filter(meds_grouped.before_or_after_index == "before") 
    meds_during = meds_grouped.filter(meds_grouped.before_or_after_index == "during")
    meds_after  = meds_grouped.filter(meds_grouped.before_or_after_index == "after")

    # Count up the number of each medication feature
    meds_counts_before = meds_before.groupBy(["person_id", "before_or_after_index"]).pivot("feature_name").agg(F.first("med_count"))
    meds_counts_during = meds_during.groupBy(["person_id", "before_or_after_index"]).pivot("feature_name").agg(F.first("med_count"))
    meds_counts_after  = meds_after.groupBy(["person_id", "before_or_after_index"]).pivot("feature_name").agg(F.first("med_count"))

    # Change column names to add prefix using global function rename_cols()
    meds_counts_before = rename_cols(meds_counts_before, suffix="_before")
    meds_counts_during = rename_cols(meds_counts_during, suffix="_during")
    meds_counts_after  = rename_cols(meds_counts_after,  suffix="_after")

    # Outer join the 3 together on person_id
    meds_df = meds_counts_before\
                .join(meds_counts_during, on=("person_id"), how="outer")\
                .join(meds_counts_after,  on=("person_id"), how="outer")
    
    # NA is interpreted as no instance of that med
    meds_df = meds_df.fillna(0)

    return meds_df


def get_meds_dataset(concept_set_members, drug_exposure, index_range):
    """
    The "Public" function, meant to be called from the main data preparation script

    Returns formatted meds features
    """
    
    pass