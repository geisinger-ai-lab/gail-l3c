
def get_smoking_concepts(concept_set_members):
    sql = """SELECT *, 'smoker' as feature_name
FROM concept_set_members
where codeset_id = 628969102 -- TOBACCO SMOKER (v2)
"""

def get_observation(obs_path):
    """
    columns:  
    person_id,
    measurement_id,
    measurement_date,
    measurement_datetime,
    measurement_time,
    value_as_number,
    range_low,
    range_high,
    provider_id,
    visit_occurrence_id,
    visit_detail_id,
    unit_source_value,
    data_partner_id,
    value_source_value,
    measurement_source_value,
    measurement_concept_id,
    measurement_type_concept_id,
    operator_concept_id,
    value_as_concept_id,
    unit_concept_id,
    measurement_source_concept_id,
    measurement_concept_name,
    measurement_type_concept_name,
    operator_concept_name,
    value_as_concept_name,
    unit_concept_name,
    measurement_source_concept_name,
    unit_concept_id_or_inferred_unit_concept_id,
    harmonized_unit_concept_id,
    harmonized_value_as_number
    """
    pass

def get_smoking_observations(observation, smoking_concepts):
    sql = """SELECT o.person_id, o.observation_date, o.observation_concept_id, o.observation_concept_name, s.feature_name
FROM observation_merge o
left join smoking_concepts s on s.concept_id = o.observation_concept_id
where s.feature_name is not null
"""

def smoking_status_dataset(smoking_observations):
    sql = """
SELECT distinct person_id, 1 as smoker
FROM smoking_observations
where feature_name = 'smoker'"""


def get_smoking_status_dataset(concept_set_members, observation):
    """
    Public function to get smoking features from raw tables
    """

    smoking_concepts = get_smoking_concepts(concept_set_members)
    smoking_observations = get_smoking_observations(observation, smoking_concepts)
    return smoking_status_dataset(smoking_observations)