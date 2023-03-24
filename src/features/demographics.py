
def get_person(person_path):
    """
    columns: 
    person_id,
    year_of_birth,
    month_of_birth,
    day_of_birth,
    birth_datetime,
    location_id,
    provider_id,
    care_site_id,
    person_source_value,
    data_partner_id,
    gender_source_value,
    race_source_value,
    ethnicity_source_value,
    gender_concept_id,
    race_concept_id,
    ethnicity_concept_id,
    gender_source_concept_id,
    race_source_concept_id,
    ethnicity_source_concept_id,
    gender_concept_name,
    race_concept_name,
    ethnicity_concept_name,
    gender_source_concept_name,
    race_source_concept_name,
    ethnicity_source_concept_name,
    is_age_90_or_older
    """
    pass


def person_demographics( person ):
    df1 = person

    from pyspark.ml.feature import StringIndexer
    df = df1.select("person_id", "gender_concept_name", "race_concept_name", "ethnicity_concept_name", "year_of_birth", "isTrainSet")
    
    df_gender = df.groupBy("person_id", "year_of_birth").pivot("gender_concept_name").count() \
        .select("person_id","year_of_birth", "FEMALE", "MALE")

    df_race = df.groupBy("person_id", "year_of_birth").pivot("race_concept_name").count() \
        .select("person_id", "year_of_birth", "Asian", "Asian Indian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "White")

    df_ethnicity = df.groupBy("person_id", "year_of_birth").pivot("ethnicity_concept_name").count() \
        .select("person_id", "year_of_birth", "Hispanic or Latino", "Not Hispanic or Latino")

    df = df1["person_id", "year_of_birth", "isTrainSet"].join(df_race, on=["person_id", "year_of_birth"], how='left') \
                                                        .join(df_ethnicity, on=["person_id", "year_of_birth"], how='left') \
                                                        .join(df_gender, on=["person_id", "year_of_birth"], how='left') \
                                                        .withColumn("age", F.lit(2022) - F.col("year_of_birth")) \
                                                        .drop("year_of_birth") \
                                                        .fillna(0)

    for col in df.columns:
        df = df.withColumnRenamed(col, col.replace(" ", "_"))
    
    return df

