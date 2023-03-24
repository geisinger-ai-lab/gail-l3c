

# Long Covid Silver Standard
"""
SELECT person_id, covid_index, pasc_code_after_four_weeks, pasc_code_prior_four_weeks, time_to_pasc
FROM Long_COVID_Silver_Standard_train
UNION ALL
SELECT person_id, covid_index, NULL as pasc_code_after_four_weeks, NULL as pasc_code_prior_four_weeks, NULL as time_to_pasc
FROM Long_COVID_Silver_Standard_Blinded_test
"""

# microvisits to macrovisits
"""
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
"""


# Index range function
"""
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
"""