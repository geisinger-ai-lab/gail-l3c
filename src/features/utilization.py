import pandas as pd
from pyspark.sql import types as T

from src.common import get_spark_session


def get_utilization():
    spark = get_spark_session()

    schema = T.StructType(
        [
            T.StructField("person_id", T.StringType()),
            T.StructField("is_index_ed", T.IntegerType()),
            T.StructField("is_index_ip", T.IntegerType()),
            T.StructField("is_index_tele", T.IntegerType()),
            T.StructField("is_index_op", T.IntegerType()),
            T.StructField("avg_los", T.DoubleType()),
            T.StructField("avg_icu_los", T.DoubleType()),
            T.StructField("before_ed_cnt", T.LongType()),
            T.StructField("before_ip_cnt", T.LongType()),
            T.StructField("before_op_cnt", T.LongType()),
            T.StructField("during_ed_cnt", T.LongType()),
            T.StructField("during_ip_cnt", T.LongType()),
            T.StructField("during_op_cnt", T.LongType()),
            T.StructField("after_ed_cnt", T.LongType()),
            T.StructField("after_ip_cnt", T.LongType()),
            T.StructField("after_op_cnt", T.LongType()),
        ]
    )
    # data = [["1"] + [1] * 4 + [1.0] * 2 + [1] * 9]
    data = []
    return spark.createDataFrame(data, schema=schema)


# Add_ICU (d5691458-1e67-4887-a68f-4cfbd4753295): v1


def add_icu(
    microvisits_to_macrovisits_merge,
    concept_set_members,
    procedure_occurrence_merge,
    condition_occurrence_merge,
    observation_merge,
):
    icu_codeset_id = 469361388

    icu_concepts = concept_set_members.filter(F.col("codeset_id") == icu_codeset_id).select(
        "concept_id", "concept_name"
    )

    procedures_df = procedure_occurrence_merge[["visit_occurrence_id", "procedure_concept_id"]]
    condition_df = condition_occurrence_merge[["visit_occurrence_id", "condition_concept_id"]]
    observation_df = observation_merge[["visit_occurrence_id", "observation_concept_id"]]

    df = microvisits_to_macrovisits_merge

    df = with_concept_name(
        df,
        procedures_df,
        icu_concepts,
        "procedure_concept_id",
        "procedure_concept_name",
    )
    df = with_concept_name(
        df, condition_df, icu_concepts, "condition_concept_id", "condition_concept_name"
    )
    df = with_concept_name(
        df,
        observation_df,
        icu_concepts,
        "observation_concept_id",
        "observation_concept_name",
    )

    df = df.withColumn(
        "is_icu",
        F.when(
            F.coalesce(
                df["procedure_concept_name"],
                df["condition_concept_name"],
                df["observation_concept_name"],
            ).isNotNull(),
            1,
        ).otherwise(0),
    )

    return df


# -- Add LOS and COVID Index (016fc630-b7c6-405e-b4a8-7c5bbb03dfef): v1
"""SELECT icu.*, s.covid_index, 
coalesce(icu.macrovisit_start_date, icu.visit_start_date) stay_start_date,
coalesce(icu.macrovisit_end_date, icu.visit_end_date) stay_end_date, 
case when 
coalesce(icu.macrovisit_end_date, icu.visit_end_date) is not null then
datediff(coalesce(icu.macrovisit_end_date, icu.visit_end_date), coalesce(icu.macrovisit_start_date, icu.visit_start_date)) 
else 0 end los
FROM add_icu icu 
left join Long_COVID_Silver_Standard s on icu.person_id = s.person_id
"""


def coerce_los_outliers(add_los_and_index):
    df = add_los_and_index

    if COERCE_LOS_OUTLIERS:
        df = df.withColumn(
            "los_mod",
            F.when(F.col("los") > LOS_MAX, 0).when(F.col("los") < 0, 0).otherwise(F.col("los")),
        )
        df = df.drop("los").withColumnRenamed("los_mod", "los")

    return df


"""SELECT *, 
case when visit_concept_name like '%Emergency%' then 1 else 0 end is_ed,
case when visit_concept_name like '%Inpatient%' then 1 else 0 end is_ip,
case when visit_concept_name like '%Tele%' then 1 else 0 end is_tele,
case when visit_concept_name not like '%Emergency%' 
    and visit_concept_name not like '%Inpatient%' 
    and visit_concept_name not like '%Tele%' then 1 else 0 end is_op
FROM coerce_los_outliers
"""


def before_index_visit_name_counts_copied(add_ed_ip_op_copied, index_range):
    idx_df = index_range.select("person_id", "index_start_date", "index_end_date")
    add_ed_ip_op = add_ed_ip_op_copied
    df = add_ed_ip_op.join(idx_df, "person_id", how="left")
    before_df = df.where(
        F.coalesce(F.col("visit_end_date"), F.col("visit_start_date")) < F.col("index_start_date")
    )

    counts_df = before_df.groupBy("person_id").agg(
        F.sum("is_ed").alias("before_ed_cnt"),
        F.sum("is_ip").alias("before_ip_cnt"),
        F.sum("is_op").alias("before_op_cnt"),
        F.sum("is_tele").alias("before_tele_cnt"),
    )

    return counts_df


def during_index_visit_name_counts_copied(add_ed_ip_op_copied, index_range):
    idx_df = index_range.select("person_id", "index_start_date", "index_end_date")
    add_ed_ip_op = add_ed_ip_op_copied
    df = add_ed_ip_op.join(idx_df, "person_id", how="left")

    during_df = df.where(
        (F.col("visit_start_date") >= F.col("index_start_date"))
        & (
            F.coalesce(F.col("visit_end_date"), F.col("visit_start_date"))
            <= F.col("index_end_date")
        )
    )

    counts_df = during_df.groupBy("person_id").agg(
        F.sum("is_ed").alias("during_ed_cnt"),
        F.sum("is_ip").alias("during_ip_cnt"),
        F.sum("is_op").alias("during_op_cnt"),
        F.sum("is_tele").alias("during_tele_cnt"),
    )

    return counts_df


def after_index_visit_name_counts_copied(add_ed_ip_op_copied, index_range):
    idx_df = index_range.select("person_id", "index_start_date", "index_end_date")
    add_ed_ip_op = add_ed_ip_op_copied
    df = add_ed_ip_op.join(idx_df, "person_id", how="left")
    during_df = df.where(F.col("visit_start_date") > F.col("index_end_date"))

    counts_df = during_df.groupBy("person_id").agg(
        F.sum("is_ed").alias("after_ed_cnt"),
        F.sum("is_ip").alias("after_ip_cnt"),
        F.sum("is_op").alias("after_op_cnt"),
        F.sum("is_tele").alias("after_tele_cnt"),
    )

    return counts_df


# index_visit_concept_name_copied
"""select 
c.person_id, 
case when any(c.visit_concept_name like '%Emergency%') then 1 else 0 end is_index_ed,
case when any(c.visit_concept_name like '%Inpatient%') then 1 else 0 end is_index_ip,
case when any(c.visit_concept_name like '%Tele%') then 1 else 0 end is_index_tele,
case when (any(c.visit_concept_name not like '%Emergency%')
        and any(c.visit_concept_name not like '%Inpatient%')
        and any(c.visit_concept_name not like '%Tele%')) then 1 else 0 end is_index_op
from
(SELECT distinct i.person_id, 
v.visit_concept_name
FROM index_range i
left join add_ed_ip_op_copied v on i.person_id = v.person_id) c
group by c.person_id

-- select * from 
-- (SELECT distinct i.person_id, 
-- v.visit_concept_name, 
-- row_number() over (partition by i.person_id order by v.visit_concept_name desc) rn
-- FROM index_range i
-- left join add_ed_ip_op_copied v on i.person_id = v.person_id) c 
-- where rn = 1"""

# los_stats_copied
"""select overall.person_id, overall.avg_los, 
case when icu.avg_icu_los is not null then icu.avg_icu_los else 0 end avg_icu_los from 
(select person_id, avg(los) avg_los from 
(select distinct person_id, stay_start_date, stay_end_date, los from add_ed_ip_op_copied ) o 
group by person_id) overall 
left join 
(select person_id, avg(los) avg_icu_los from 
(select distinct person_id, stay_start_date, stay_end_date, los from add_ed_ip_op_copied 
where is_icu = 1) o 
group by person_id) icu
on icu.person_id = overall.person_id
"""

# utilization_v2
"""
SELECT 
i.person_id,
i.is_index_ed,
i.is_index_ip,
i.is_index_tele,
i.is_index_op,
l.avg_los,
l.avg_icu_los,
b.before_ed_cnt,
b.before_ip_cnt,
b.before_op_cnt,
d.during_ed_cnt,
d.during_ip_cnt,
d.during_op_cnt,
a.after_ed_cnt,
a.after_ip_cnt,
a.after_op_cnt
FROM index_visit_concept_name_copied i
left join los_stats_copied l on i.person_id = l.person_id
left join before_index_visit_name_counts_copied b on i.person_id = b.person_id
left join during_index_visit_name_counts_copied d on i.person_id = d.person_id
left join after_index_visit_name_counts_copied a on i.person_id = a.person_id
"""


def utilization_updated_columns(utilization_v2):
    df = utilization_v2
    subset_fill_0 = [
        "is_index_ed",
        "is_index_ip",
        "is_index_tele",
        "is_index_op",
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
    subset_fill_neg_1 = ["avg_los", "avg_icu_los"]
    df = df.fillna(0, subset=subset_fill_0)
    df = df.fillna(-1, subset=subset_fill_neg_1)
    return df
