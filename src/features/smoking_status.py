from src.common import get_spark_session

spark = get_spark_session()


def get_smoking_concepts(concept_set_members):
    concept_set_members.createOrReplaceTempView("concept_set_members")

    sql = """SELECT *, 'smoker' as feature_name
FROM concept_set_members
where codeset_id = 628969102 -- TOBACCO SMOKER (v2)
"""
    return spark.sql(sql)


def get_smoking_observations(observation, smoking_concepts):
    observation.createOrReplaceTempView("observation")
    smoking_concepts.createOrReplaceTempView("smoking_concepts")

    sql = """SELECT 
    o.person_id, 
    o.observation_date, 
    o.observation_concept_id, 
    s.feature_name
FROM observation o
left join smoking_concepts s on s.concept_id = o.observation_concept_id
where s.feature_name is not null
"""

    return spark.sql(sql)


def smoking_status_dataset(smoking_observations):
    smoking_observations.createOrReplaceTempView("smoking_observations")

    sql = """
SELECT distinct person_id, 1 as smoker
FROM smoking_observations
where feature_name = 'smoker'"""

    return spark.sql(sql)


def get_smoking_status_dataset(concept_set_members, observation):
    """
    Public function to get smoking features from raw tables
    """

    smoking_concepts = get_smoking_concepts(concept_set_members)
    smoking_observations = get_smoking_observations(observation, smoking_concepts)
    return smoking_status_dataset(smoking_observations)


if __name__ == "__main__":
    # Load data as spark DF
    concept_set_members_path = "data/raw_sample/concept_set_members.csv"
    concept_set_members = spark.read.csv(
        concept_set_members_path, header=True, inferSchema=True
    )

    observation_path = "data/raw_sample/training/observation.csv"
    observation = spark.read.csv(observation_path, header=True, inferSchema=True)

    smoking_status = get_smoking_status_dataset(concept_set_members, observation)
    smoking_status.show()
