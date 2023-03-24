

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.53510587-c350-44d1-a09d-7a3d7408a87e"),
    Procedure_all=Input(rid="ri.foundry.main.dataset.856243d6-d7ae-44e9-8fdf-a83ea0396897"),
    covid_index_date=Input(rid="ri.foundry.main.dataset.4304253b-93e5-4b93-b702-7f95d7934f6b")
)
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1dec59e8-d4ce-473e-88db-8e4d5882b73e"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    measurement_merge=Input(rid="ri.foundry.main.dataset.fa60891b-2203-49b1-862e-7d04ff92776d")
)
SELECT LABS.person_id, 
    LABS.measurement_date, LABS.measurement_concept_name, LABS.measurement_source_concept_name,
    LABS.value_as_number, LABS.unit_concept_name, LABS.range_low, LABS.range_high
    FROM measurement_merge AS LABS
    RIGHT JOIN concept_set_members AS LOOKUP
        ON LABS.measurement_concept_id = LOOKUP.concept_id
    WHERE LOOKUP.codeset_id IN ( 985622897 ) -- Complete Blood Panel  
        AND LABS.person_id IS NOT NULL
        AND LABS.value_as_number IS NOT NULL
    ORDER BY LABS.person_id, LABS.measurement_date

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0b172844-9b39-499f-b71e-f026e35ce69d"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    measurement_merge=Input(rid="ri.foundry.main.dataset.fa60891b-2203-49b1-862e-7d04ff92776d")
)
SELECT LABS.person_id, 
    LABS.measurement_date, LABS.measurement_concept_name, LABS.measurement_source_concept_name,
    LABS.value_as_number, LABS.unit_concept_name, LABS.range_low, LABS.range_high
    FROM measurement_merge AS LABS
    RIGHT JOIN concept_set_members AS LOOKUP
        ON LABS.measurement_concept_id = LOOKUP.concept_id
    WHERE LOOKUP.codeset_id IN ( 212998332, --Comprehensive metabolic Profile
                                 104464584 --Albumin
                                 ) 
        AND LABS.person_id IS NOT NULL
        AND LABS.value_as_number IS NOT NULL
    ORDER BY LABS.person_id, LABS.measurement_date

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.232c95d8-1afb-4f60-bfbf-3c1bb7f15606"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence_merge=Input(rid="ri.foundry.main.dataset.ed6feda8-c72f-4a3e-8c6e-a97c6695d0f2")
)
SELECT *
    FROM condition_occurrence_merge as cond 
    INNER JOIN concept_set_members as set_cond 
        ON cond.condition_concept_id = set_cond.concept_id
    WHERE set_cond.codeset_id IN ('834391873', '882775108', '18918743', '248468138', '581513221', '628969102', '602584947','33199070')
    ORDER BY cond.person_id 

--  WHERE set_cond.concept_set_name IN ('HYPERTENSION', 'HEART FAILURE', 'DIABETES COMPLICATED', 'DIABETES UNCOMPLICATED', 'OBESITY', 'TOBACCO SMOKER', '[N3C][GAIL]asthma','[L3C][GAIL] COPD')
--  ORDER BY cond.person_id 

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cd8394ac-1a63-41f3-a52c-eef5c28ffe57"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    measurement_merge=Input(rid="ri.foundry.main.dataset.fa60891b-2203-49b1-862e-7d04ff92776d")
)
SELECT LABS.person_id, 
    LABS.measurement_date, LABS.measurement_concept_name, LABS.measurement_source_concept_name,
    LABS.value_as_number, LABS.unit_concept_name, LABS.range_low, LABS.range_high
    FROM measurement_merge AS LABS
    RIGHT JOIN concept_set_members AS LOOKUP
        ON LABS.measurement_concept_id = LOOKUP.concept_id
    WHERE LOOKUP.codeset_id IN ( 459475527 ) -- ImmunoAssay 
        AND LABS.person_id IS NOT NULL
        AND LABS.value_as_number IS NOT NULL
    ORDER BY LABS.person_id, LABS.measurement_date

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2eeb4595-97b9-45a3-9430-6b76e738a3e0"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    measurement_merge=Input(rid="ri.foundry.main.dataset.fa60891b-2203-49b1-862e-7d04ff92776d")
)
SELECT LABS.person_id, 
    LABS.measurement_date, LABS.measurement_concept_name, LABS.measurement_source_concept_name,
    LABS.value_as_number, LABS.unit_concept_name, LABS.range_low, LABS.range_high
    FROM measurement_merge AS LABS
    RIGHT JOIN concept_set_members AS LOOKUP
        ON LABS.measurement_concept_id = LOOKUP.concept_id
    WHERE LOOKUP.codeset_id IN ( 
                                 854721978, --BodyWeight
                                 186671483 --Height
                                 ) 
        AND LABS.person_id IS NOT NULL
        AND LABS.value_as_number IS NOT NULL
    ORDER BY LABS.person_id, LABS.measurement_date

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cbdbafa4-b89f-4559-b704-3122e5c4b8c1"),
    Long_COVID_Silver_Standard_Blinded_test=Input(rid="ri.foundry.main.dataset.cb65632b-bdff-4aa9-8696-91bc6667e2ba"),
    Long_COVID_Silver_Standard_train=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671")
)
SELECT person_id, covid_index, pasc_code_after_four_weeks, pasc_code_prior_four_weeks, time_to_pasc
FROM Long_COVID_Silver_Standard_train
UNION ALL
SELECT person_id, covid_index, NULL as pasc_code_after_four_weeks, NULL as pasc_code_prior_four_weeks, NULL as time_to_pasc
FROM Long_COVID_Silver_Standard_Blinded_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.856243d6-d7ae-44e9-8fdf-a83ea0396897"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    procedure_occurrence_merge=Input(rid="ri.foundry.main.dataset.44ffaa66-95d6-4323-817d-4bbf1bf84f12")
)
SELECT *
    FROM procedure_occurrence_merge as proce
    INNER JOIN concept_set_members as proc_concpt
        ON proce.procedure_concept_id = proc_concpt.concept_id
    WHERE proc_concpt.codeset_id  IN ('469361388', '629960249', '260036299', '838273021','415149730', '850548108', '129959605','429864170' )
    ORDER BY proce.person_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c82ed648-a060-4559-87b8-30d3d03065a9"),
    coerce_los_outliers=Input(rid="ri.foundry.main.dataset.9605effb-6651-49ba-969d-8c79bda5bce6")
)
SELECT *, 
case when visit_concept_name like '%Emergency%' then 1 else 0 end is_ed,
case when visit_concept_name like '%Inpatient%' then 1 else 0 end is_ip,
case when visit_concept_name like '%Tele%' then 1 else 0 end is_tele,
case when visit_concept_name not like '%Emergency%' 
    and visit_concept_name not like '%Inpatient%' 
    and visit_concept_name not like '%Tele%' then 1 else 0 end is_op
FROM coerce_los_outliers

@transform_pandas(
    Output(rid="ri.vector.main.execute.a8e8ce77-0c4b-42c2-92b9-c30c3f90d14b"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.cbdbafa4-b89f-4559-b704-3122e5c4b8c1"),
    add_icu=Input(rid="ri.vector.main.execute.1ba45b78-bb25-4ba7-9baf-755acd57d460")
)
-- Add LOS and COVID Index (016fc630-b7c6-405e-b4a8-7c5bbb03dfef): v1
SELECT icu.*, s.covid_index, 
coalesce(icu.macrovisit_start_date, icu.visit_start_date) stay_start_date,
coalesce(icu.macrovisit_end_date, icu.visit_end_date) stay_end_date, 
case when 
coalesce(icu.macrovisit_end_date, icu.visit_end_date) is not null then
datediff(coalesce(icu.macrovisit_end_date, icu.visit_end_date), coalesce(icu.macrovisit_start_date, icu.visit_start_date)) 
else 0 end los
FROM add_icu icu 
left join Long_COVID_Silver_Standard s on icu.person_id = s.person_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ed6feda8-c72f-4a3e-8c6e-a97c6695d0f2"),
    condition_occurrence_test=Input(rid="ri.foundry.main.dataset.3e01546f-f110-4c67-a6db-9063d2939a74"),
    condition_occurrence_train=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2")
)
SELECT 
person_id,
condition_occurrence_id,
condition_end_date,
condition_end_datetime,
condition_start_date,
condition_start_datetime,
data_partner_id,
provider_id,
stop_reason,
visit_detail_id,
visit_occurrence_id,
condition_source_value,
condition_status_source_value,
condition_concept_id,
condition_source_concept_id,
condition_status_concept_id,
condition_type_concept_id,
condition_concept_name,
condition_source_concept_name,
condition_status_concept_name,
condition_type_concept_name
FROM condition_occurrence_train
UNION ALL
SELECT 
person_id,
condition_occurrence_id,
condition_end_date,
condition_end_datetime,
condition_start_date,
condition_start_datetime,
data_partner_id,
provider_id,
stop_reason,
visit_detail_id,
visit_occurrence_id,
condition_source_value,
condition_status_source_value,
condition_concept_id,
condition_source_concept_id,
condition_status_concept_id,
condition_type_concept_id,
condition_concept_name,
condition_source_concept_name,
condition_status_concept_name,
condition_type_concept_name
FROM condition_occurrence_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4304253b-93e5-4b93-b702-7f95d7934f6b"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.cbdbafa4-b89f-4559-b704-3122e5c4b8c1"),
    microvisits_to_macrovisits_merge=Input(rid="ri.foundry.main.dataset.fa1a8bfa-4e83-4fb8-9928-13378f987820")
)
-- from Grants shared workbook
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d78c602f-06a3-410e-9092-fbd66f0091bb"),
    drug_exposure_test=Input(rid="ri.foundry.main.dataset.26a51cab-0279-45a6-bbc0-f44a12b52f9c"),
    drug_exposure_train=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5")
)
SELECT 
person_id,
drug_exposure_id,
data_partner_id,
days_supply,
dose_unit_source_value,
drug_exposure_end_date,
drug_exposure_end_datetime,
drug_exposure_start_date,
drug_exposure_start_datetime,
drug_source_value,
lot_number,
provider_id,
quantity,
refills,
route_source_value,
sig,
stop_reason,
verbatim_end_date,
visit_detail_id,
visit_occurrence_id,
drug_concept_id,
drug_source_concept_id,
drug_type_concept_id,
route_concept_id,
drug_concept_name,
drug_source_concept_name,
drug_type_concept_name,
route_concept_name
FROM drug_exposure_train
UNION ALL 
SELECT 
person_id,
drug_exposure_id,
data_partner_id,
days_supply,
dose_unit_source_value,
drug_exposure_end_date,
drug_exposure_end_datetime,
drug_exposure_start_date,
drug_exposure_start_datetime,
drug_source_value,
lot_number,
provider_id,
quantity,
refills,
route_source_value,
sig,
stop_reason,
verbatim_end_date,
visit_detail_id,
visit_occurrence_id,
drug_concept_id,
drug_source_concept_id,
drug_type_concept_id,
route_concept_id,
drug_concept_name,
drug_source_concept_name,
drug_type_concept_name,
route_concept_name
FROM drug_exposure_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.37168eef-4434-4de4-9fca-2d1ab449e3b2"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.cbdbafa4-b89f-4559-b704-3122e5c4b8c1"),
    microvisits_to_macrovisits_merge=Input(rid="ri.foundry.main.dataset.fa1a8bfa-4e83-4fb8-9928-13378f987820")
)
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ab4f364d-f2ab-4ff1-90ab-69d71918b490"),
    add_ed_ip_op_copied=Input(rid="ri.foundry.main.dataset.c82ed648-a060-4559-87b8-30d3d03065a9"),
    index_range=Input(rid="ri.foundry.main.dataset.37168eef-4434-4de4-9fca-2d1ab449e3b2")
)
select 
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
-- where rn = 1

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d3a76a0f-f803-4f3a-ade3-3aa70ef078de"),
    add_ed_ip_op_copied=Input(rid="ri.foundry.main.dataset.c82ed648-a060-4559-87b8-30d3d03065a9")
)
select overall.person_id, overall.avg_los, 
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fa60891b-2203-49b1-862e-7d04ff92776d"),
    measurement_test=Input(rid="ri.foundry.main.dataset.b7749e49-cf01-4d0a-a154-2f00eecab21e"),
    measurement_train=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7")
)
SELECT
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
FROM measurement_train
UNION ALL
SELECT 
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
FROM measurement_test


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fa1a8bfa-4e83-4fb8-9928-13378f987820"),
    microvisits_to_macrovisits_test=Input(rid="ri.foundry.main.dataset.f5008fa4-e736-4244-88e1-1da7a68efcdb"),
    microvisits_to_macrovisits_train=Input(rid="ri.foundry.main.dataset.d77a701f-34df-48a1-a71c-b28112a07ffa")
)
SELECT
    person_id,
visit_occurrence_id,
care_site_id,
data_partner_id,
preceding_visit_occurrence_id,
provider_id,
visit_end_date,
visit_end_datetime,
visit_start_date,
visit_start_datetime,
visit_source_value,
admitting_source_value,
discharge_to_source_value,
admitting_source_concept_id,
discharge_to_concept_id,
visit_concept_id,
visit_source_concept_id,
visit_type_concept_id,
admitting_source_concept_name,
discharge_to_concept_name,
visit_concept_name,
visit_source_concept_name,
visit_type_concept_name,
macrovisit_id,
macrovisit_start_date,
macrovisit_end_date
FROM microvisits_to_macrovisits_train
UNION ALL 
SELECT 
person_id,
visit_occurrence_id,
care_site_id,
data_partner_id,
preceding_visit_occurrence_id,
provider_id,
visit_end_date,
visit_end_datetime,
visit_start_date,
visit_start_datetime,
visit_source_value,
admitting_source_value,
discharge_to_source_value,
admitting_source_concept_id,
discharge_to_concept_id,
visit_concept_id,
visit_source_concept_id,
visit_type_concept_id,
admitting_source_concept_name,
discharge_to_concept_name,
visit_concept_name,
visit_source_concept_name,
visit_type_concept_name,
macrovisit_id,
macrovisit_start_date,
macrovisit_end_date
FROM microvisits_to_macrovisits_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.67d6b203-2db2-4289-a12b-027fe2b52f22"),
    observation_test=Input(rid="ri.foundry.main.dataset.fc1ce22e-9cf6-4335-8ca7-aa8c733d506d"),
    observation_train=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516")
)
SELECT 
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
FROM observation_train
UNION ALL 
SELECT 
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
FROM observation_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.18101d6a-e393-4681-b03a-a4ffdadc90b2"),
    person_test=Input(rid="ri.foundry.main.dataset.06629068-25fc-4802-9b31-ead4ed515da4"),
    person_train=Input(rid="ri.foundry.main.dataset.f71ffe18-6969-4a24-b81c-0e06a1ae9316")
)
SELECT 
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
is_age_90_or_older, 
1 as isTrainSet
FROM person_train
UNION ALL
SELECT 
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
is_age_90_or_older, 
0 as isTrainSet 
FROM person_test

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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dbcfa7b1-6aa2-43ee-bb81-9bdb3dd02695"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
SELECT *, 'smoker' as feature_name
FROM concept_set_members
where codeset_id = 628969102 -- TOBACCO SMOKER (v2)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.34d2fd38-d03f-4f08-8a99-bd6617c828be"),
    smoking_observations=Input(rid="ri.foundry.main.dataset.310c5d45-ab13-4dfb-809c-96f173f5212d")
)
SELECT distinct person_id, 1 as smoker
FROM smoking_observations
where feature_name = 'smoker'

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.310c5d45-ab13-4dfb-809c-96f173f5212d"),
    observation_merge=Input(rid="ri.foundry.main.dataset.67d6b203-2db2-4289-a12b-027fe2b52f22"),
    smoking_concepts=Input(rid="ri.foundry.main.dataset.dbcfa7b1-6aa2-43ee-bb81-9bdb3dd02695")
)
SELECT o.person_id, o.observation_date, o.observation_concept_id, o.observation_concept_name, s.feature_name
FROM observation_merge o
left join smoking_concepts s on s.concept_id = o.observation_concept_id
where s.feature_name is not null

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.07b884d2-c5d8-42c1-af45-2127206c99e6"),
    after_index_visit_name_counts_copied=Input(rid="ri.foundry.main.dataset.bb137691-7bd6-4df6-adf6-71efebaea2c5"),
    before_index_visit_name_counts_copied=Input(rid="ri.foundry.main.dataset.9b29b83d-2db0-4e05-99cd-ac844d026435"),
    during_index_visit_name_counts_copied=Input(rid="ri.foundry.main.dataset.0b095fdd-ee5e-4864-9ee7-8a351c4708af"),
    index_visit_concept_name_copied=Input(rid="ri.foundry.main.dataset.ab4f364d-f2ab-4ff1-90ab-69d71918b490"),
    los_stats_copied=Input(rid="ri.foundry.main.dataset.d3a76a0f-f803-4f3a-ade3-3aa70ef078de")
)
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.190e00c4-8de1-4864-b768-95af1c16c59e"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
select *, 'spo2' as feature_name, 'percent' as preferred_unit_concept_name
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b199d167-0207-445f-8b2e-8b19fefad768"),
    index_range=Input(rid="ri.foundry.main.dataset.37168eef-4434-4de4-9fca-2d1ab449e3b2"),
    measurement_merge=Input(rid="ri.foundry.main.dataset.fa60891b-2203-49b1-862e-7d04ff92776d"),
    vitals_concepts=Input(rid="ri.foundry.main.dataset.190e00c4-8de1-4864-b768-95af1c16c59e")
)
select 
    m.person_id
    , m.measurement_date
    , m.measurement_concept_id
    , m.measurement_concept_name
    , m.value_as_number
    , m.value_as_concept_id
    , m.value_as_concept_name
    , m.unit_concept_id
    , m.unit_concept_name
    , m.measurement_source_concept_id
    , m.measurement_source_concept_name
    , m.range_low
    , m.range_high
    , m.data_partner_id
    , case when (m.measurement_date >= ci.index_start_date) and (m.measurement_date <= ci.index_end_date) then 'during'
        when m.measurement_date < ci.index_start_date then 'before'
        when m.measurement_date > ci.index_end_date then 'after'
    end as before_or_after_index
    , cs.feature_name
from        measurement_merge                   as m
left join   index_range                         as ci on ci.person_id = m.person_id
left join   vitals_concepts                     as cs on m.measurement_concept_id = cs.concept_id
where 
    cs.feature_name is not null
    and m.value_as_number is not null

    -- Remove extreme values for vitals:
    and m.value_as_number >=0
    and m.value_as_number < 900 
    
    -- Match preferred united when specified
    and (cs.preferred_unit_concept_name is null or cs.preferred_unit_concept_name = m.unit_concept_name) 
;

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b22f0cc5-51b8-4e5d-a46a-c4dff3a0f5f8"),
    vitals_filtered=Input(rid="ri.foundry.main.dataset.b199d167-0207-445f-8b2e-8b19fefad768")
)
select 
    v.person_id
    , v.feature_name
    , v.before_or_after_index
    , avg(v.value_as_number) as vital_avg
    , stddev(v.value_as_number) as vital_stddev 
    , count(v.value_as_number) as vital_count
from vitals_filtered v
group by v.person_id, v.feature_name, v.before_or_after_index

