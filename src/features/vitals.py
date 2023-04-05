"""
function to load data (concepts, measurement, index_range(derrived))

Implement the sql steps as functions that take in dataframes

"public" function that returns the formatted med features, 
    with path parameters (e.g. train/test)

"""

from pyspark.sql import functions as F

from src.common import get_spark_session, rename_cols

spark = get_spark_session()


def get_vitals_concepts(concept_set_members):

    concept_set_members.createOrReplaceTempView("concept_set_members")

    sql = """select *, 'spo2' as feature_name, 'percent' as preferred_unit_concept_name
from concept_set_members csm
where csm.codeset_id = 298033057 -- [Feasibility] Oxygen saturation
union
select *, 'bp_systolic' as feature_name, 'millimeter mercury column' as preferred_unit_concept_name
from concept_set_members csm
where csm.codeset_id = 963705854 -- Systolic blood pressure
union
select *, 'bp_diastolic' as feature_name, 'millimeter mercury column' as preferred_unit_concept_name
from concept_set_members csm
where csm.codeset_id = 573275931 -- Diastolic Blood Pressure (LG33056-9 and Snomed)
union
select *, 'heart_rate' as feature_name, null as preferred_unit_concept_name 
from concept_set_members csm
where csm.codeset_id = 844379916 -- Heart rate (LG33055-1 and SNOMED)
union
select *, 'resp_rate' as feature_name, null as preferred_unit_concept_name 
from concept_set_members csm
where csm.codeset_id = 510707388 -- Respiratory rate (LG33055-1 and SNOMED)
;
"""
    return spark.sql(sql)


def filter_vitals(vitals_concepts, index_range, measurement):

    vitals_concepts.createOrReplaceTempView("vitals_concepts")
    index_range.createOrReplaceTempView("index_range")
    measurement.createOrReplaceTempView("measurement")

    sql = """select 
    m.person_id
    , m.measurement_date
    , m.measurement_concept_id
    , m.value_as_number
    , m.value_as_concept_id
    , m.unit_concept_id
    -- , m.unit_concept_name
    , m.measurement_source_concept_id
    , m.range_low
    , m.range_high
    , case when (m.measurement_date >= ci.index_start_date) and (m.measurement_date <= ci.index_end_date) then 'during'
        when m.measurement_date < ci.index_start_date then 'before'
        when m.measurement_date > ci.index_end_date then 'after'
    end as before_or_after_index
    , cs.feature_name
from        measurement                         as m
left join   index_range                         as ci on ci.person_id = m.person_id
left join   vitals_concepts                     as cs on m.measurement_concept_id = cs.concept_id
where 
    cs.feature_name is not null
    and m.value_as_number is not null

    -- Remove extreme values for vitals:
    and m.value_as_number >=0
    and m.value_as_number < 900 
    
    -- Match preferred united when specified
    -- and (cs.preferred_unit_concept_name is null or cs.preferred_unit_concept_name = m.unit_concept_name) 
;
"""
    # TODO: Fix preferred unit concept matching (use ID instead)
    return spark.sql(sql)


def group_vitals(vitals_filtered):

    vitals_filtered.createOrReplaceTempView("vitals_filtered")

    sql = """select 
    v.person_id
    , v.feature_name
    , v.before_or_after_index
    , avg(v.value_as_number) as vital_avg
    , stddev(v.value_as_number) as vital_stddev 
    , count(v.value_as_number) as vital_count
from vitals_filtered v
group by v.person_id, v.feature_name, v.before_or_after_index
"""

    return spark.sql(sql)


def vitals_dataset(vitals_grouped):
    # Pivot on feature_name
    vitals_pivoted = (
        vitals_grouped.groupBy(["person_id", "before_or_after_index"])
        .pivot("feature_name")
        .agg(F.first("vital_avg").alias("vital_avg"))
    )  # , F.first("vital_stddev").alias("vital_stddev"))

    # Break into 3 dfs (before, during, after)
    vitals_pivoted_before = vitals_pivoted.filter(
        vitals_pivoted.before_or_after_index == "before"
    )
    vitals_pivoted_during = vitals_pivoted.filter(
        vitals_pivoted.before_or_after_index == "during"
    )
    vitals_pivoted_after = vitals_pivoted.filter(
        vitals_pivoted.before_or_after_index == "after"
    )

    # Change column names to add prefix using global function rename_cols()
    vitals_pivoted_before = rename_cols(vitals_pivoted_before, suffix="_before")
    vitals_pivoted_during = rename_cols(vitals_pivoted_during, suffix="_during")
    vitals_pivoted_after = rename_cols(vitals_pivoted_after, suffix="_after")

    # Outer join the 3 together on person_id
    vitals_df = vitals_pivoted_before.join(
        vitals_pivoted_during, on=("person_id"), how="outer"
    ).join(vitals_pivoted_after, on=("person_id"), how="outer")

    return vitals_df


def get_vitals_dataset(concept_set_members, measurement, index_range):
    """
    The "Public" function, meant to be called from the main data preparation script

    Returns formatted vitals features
    """

    vitals_concepts = get_vitals_concepts(concept_set_members)
    vitals_filtered = filter_vitals(vitals_concepts, index_range, measurement)
    vitals_grouped = group_vitals(vitals_filtered)
    vitals_df = vitals_dataset(vitals_grouped)

    return vitals_df


if __name__ == "__main__":

    # Load data as spark DF
    concept_set_path = "data/raw_sample/concept_set_members.csv"
    concept_set_members = spark.read.csv(
        concept_set_path, header=True, inferSchema=True
    )
    # concept_set_members = concept_set_members.withColumn("concept_id", concept_set_members.concept_id.cast("int"))

    index_range_path = "data/intermediate/training/index_range.csv"
    index_range = spark.read.csv(index_range_path, header=True, inferSchema=True)

    measurement_path = "data/raw_sample/training/measurement.csv"
    measurement = spark.read.csv(measurement_path, header=True, inferSchema=True)

    # measurement = measurement.withColumn(
    #     "measurement_concept_id", measurement.measurement_concept_id.cast("int")
    # ).withColumn("person_id", measurement.person_id.cast("int"))
    # measurement.show()

    # observation = spark.read.csv("data/raw_sample/training/observation.csv", header=True)

    # vitals_concepts = get_vitals_concepts(concept_set_members)
    # vitals_concepts.show()

    # vitals_filtered = filter_vitals(vitals_concepts, index_range, measurement)
    # vitals_filtered.show()
    # import pdb; pdb.set_trace()

    # vitals_grouped = group_vitals(vitals_filtered)
    # vitals_grouped.show()

    # vitals_df = vitals_dataset(vitals_grouped)

    # Run the vitals data ETL
    vitals_df = get_vitals_dataset(concept_set_members, measurement, index_range)

    vitals_df.show()
