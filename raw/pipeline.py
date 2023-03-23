## Basic Libraries:
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

## PySpark Libraries:
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, create_map, lit, when 
from itertools import chain

## ML Libraries:
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplitModel

# Foundry ML Lib:
from foundry_ml import Model, Stage
from foundry_ml_sklearn.utils import extract_matrix

## Sklearn Libraries:
from sklearn.model_selection import train_test_split

## Metric / Eval Libraries:
from foundry_ml_metrics import MetricSet
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay, 
    f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve,  
    roc_auc_score, roc_curve, brier_score_loss)
from sklearn.calibration import calibration_curve


## Demographics Features: 
feat_demo = ['age', 'FEMALE', 'MALE', 
            'Asian', 'Asian_Indian', 'Black_or_African_American', 
            'Native_Hawaiian_or_Other_Pacific_Islander', 'White'] #

## Measurement Features: 
feat_meas = ['body_height_value', 'body_weight_value', 
            'alanine_aminotransferase_value', 'albumin_value', 'albumin_bcg_value', 'albumin_bcp_value', 'albumin_electrophoresis_value', 'albumin_globulin_ratio_value', 'alkaline_phosphatase_value', 'anion_gap_value', 'aspartate_aminotransferase_value', 'bicarbonate_value', 'bilirubin_total_value', 'bun_value', 'bun_creatinine_ratio_value', 'calcium_value', 'carbon_dioxide_total_value', 'chloride_value', 'creatinine_value', 'globulin_value', 'glomerular_filt_CKD_value', 'glomerular_filt_blacks_CKD_value', 'glomerular_filt_blacks_MDRD_value', 'glomerular_filt_females_MDRD_value', 'glomerular_filt_nonblacks_CKD_value', 'glomerular_filt_nonblacks_MDRD_value', 'glucose_value', 'potassium_value', 'protein_value', 'sodium_value',
             'absolute_band_neutrophils_value', 'absolute_basophils_value', 'absolute_eosinophils_value', 'absolute_lymph_value', 'absolute_monocytes_value', 'absolute_neutrophils_value', 'absolute_other_value', 'absolute_var_lymph_value', 'cbc_panel_value', 'hct_value', 'hgb_value', 'mch_value', 'mchc_value', 'mcv_value', 'mpv_value', 'pdw_volume_value', 'percent_band_neutrophils_value', 'percent_basophils_value', 'percent_eosinophils_value', 'percent_granulocytes_value', 'percent_lymph_value', 'percent_monocytes_value', 'percent_neutrophils_value', 'percent_other_value', 'percent_var_lymph_value', 'platelet_count_value', 'rbc_count_value', 'rdw_ratio_value', 'rdw_volume_value', 'wbc_count_value']
feat_meas_after = ['after_'+feat for feat in feat_meas]
feat_meas_before = ['before_'+feat for feat in feat_meas]
feat_meas_during = ['during_'+feat for feat in feat_meas]

## Vitals Features: 
feat_vitals = ['bp_diastolic_before', 'bp_systolic_before', 
            'heart_rate_before', 'resp_rate_before', 'spo2_before', 
            'bp_diastolic_during', 'bp_systolic_during', 
            'heart_rate_during', 'resp_rate_during', 'spo2_during', 
            'bp_diastolic_after', 'bp_systolic_after', 
            'heart_rate_after', 'resp_rate_after', 'spo2_after']

## Smoking Features: 
feat_smoke = ['smoker']

## Diagnosis Count Features:
feat_dxct = ['asthma', 'copd', 'diabetes_complicated', 'diabetes_uncomplicated', 
            'heart_failure', 'hypertension', 'obesity']

## Medication Count Features:
feat_meds = ['anticoagulants_before', 'asthma_drugs_before', 
            'antibiotics_during', 'antivirals_during', 'corticosteroids_during', 
            'iv_immunoglobulin_during', 'lopinavir_during', 'paxlovid_during', 
            'remdesivir_during', 
            'anticoagulants_after', 'asthma_drugs_after']

## Procedure Count Features:
feat_proc = [
    'after_Ventilator_used', 'after_Lungs_CT_scan', 'after_Chest_X_ray',
    'after_Lung_Ultrasound', 'after_ECMO_performed', 'after_ECG_performed',
    'after_Echocardiogram_performed', 'after_Blood_transfusion',
    'before_Ventilator_used', 'before_Lungs_CT_scan', 'before_Chest_X_ray',
    'before_Lung_Ultrasound', 'before_ECMO_performed', 'before_ECG_performed',
    'before_Echocardiogram_performed', 'before_Blood_transfusion', 
    'during_Ventilator_used', 'during_Lungs_CT_scan', 'during_Chest_X_ray', 
    'during_Lung_Ultrasound', 'during_ECMO_performed', 'during_ECG_performed', 
    'during_Echocardiogram_performed', 'during_Blood_transfusion'
    ]

## Utilization features:
feat_utl = ['is_index_ed', 'is_index_ip', 'is_index_tele', 'is_index_op', 
            'avg_los', 'avg_icu_los', 
            'before_ed_cnt', 'before_ip_cnt', 'before_op_cnt', 
            'during_ed_cnt', 'during_ip_cnt', 'during_op_cnt', 
            'after_ed_cnt', 'after_ip_cnt', 'after_op_cnt']


## Variable initialization for modeling: 
random_seed = 16

## Deciding what features to run and for what label:
features = feat_demo + feat_vitals + feat_meas_before + feat_meas_during + feat_meas_after + feat_smoke + feat_dxct +   feat_meds + feat_utl + feat_proc

label = 'pasc_code_after_four_weeks'


## Utilization 
COERCE_LOS_OUTLIERS = False
LOS_MAX = 60

## Custom Functions:

def with_concept_name(df, domain_df, icu_concepts, concept_id_column, concept_name_column):

    df = df.join(domain_df, 'visit_occurrence_id', how='left')

    df = df.join(icu_concepts, df[concept_id_column] == icu_concepts['concept_id'], how='left')

    df = df.withColumnRenamed("concept_name", concept_name_column).drop('concept_id').distinct()

    df = df.withColumn('rn', F.row_number().over(Window.partitionBy('visit_occurrence_id').orderBy(F.desc(concept_name_column))))

    df = df.filter(df['rn'] == 1)

    df = df.drop('rn')

    return df


def rename_cols(df, prefix="", suffix=""):
    """
    Helper function to rename columns by adding a prefix or a suffix
    """
    index_cols = ["person_id", "before_or_after_index"]
    select_list = [col(col_name).alias(prefix + col_name + suffix) if col_name not in index_cols else col(col_name) for col_name in df.columns]
    df = df.select(select_list).drop(col("before_or_after_index"))
    return df


def calculate_evaluation_metrics(model, data, threshold=0.5):
    """
    Function to calulate eval metrics and plots for a given model and data set

    model - an "object" input of the model you want to evaluate
    data - the data set to evaluate against. IMPORTANT: For the code to work the `data` needs to be loaded into the transform as a "Transform input" not a spark dataframe
    """

    ## Generate predictions for the input data:
    df_train = data.dataframe().toPandas()
    scores = model.transform( df_train[features] )
    scores['pred_label'] = (scores.probability_1 > threshold).astype('int')
    scores['true_label'] = df_train[label]

    ## Generating confusion matrix and all other values:
    cm = confusion_matrix( scores['true_label'], scores['pred_label'] )
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP) 

    ## Generating confusion matrix image:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No PASC', 'PASC'])
    disp.plot()
    cm_plot = plt.gcf()

    ## Generating precision, recall, f-measure scores:
    f1 = f1_score( scores['true_label'], scores['pred_label'], average='binary')
    precision = precision_score( scores['true_label'], scores['pred_label'], average='binary')
    recall = recall_score( scores['true_label'], scores['pred_label'], average='binary')

    ## Precision-Recall Curve
    p, r, thresholds = precision_recall_curve(scores['true_label'], scores['probability_1'])
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_curve = plt.gcf()

    ## ROC Curve
    fpr, tpr, thresholds = roc_curve(scores['true_label'], scores['probability_1'])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    roc = plt.gcf()

    ## Calibration plot
    prob_true, prob_pred = calibration_curve(scores['true_label'], scores['probability_1'], n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, 's-')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    calibration_plot = plt.gcf()

    ## Calibration Breir Score
    brier = brier_score_loss(scores['true_label'], scores['probability_1'])

    ## Score dist
    plt.figure()
    plt.hist(scores['probability_1'], range=(0, 1), bins=10, histtype="step")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of predicted probabilities")
    pred_dist = plt.gcf()

    ## Average precision
    average_precision = average_precision_score(scores['true_label'], scores['probability_1'])

    ## AU-ROC score:
    auroc = roc_auc_score( scores['true_label'], scores['probability_1'])

    ## Classification Report:
    cr = classification_report( scores['true_label'], scores['pred_label'], target_names=['No PASC', 'PASC'] )

    # Initialize MetricSet container
    metric_set = MetricSet(
        model = model,
        input_data=data
    )

    metric_set.add(name='confusion_matrix', value=cm_plot)
    metric_set.add(name=f'f1_score@{threshold}', value=f1)
    metric_set.add(name=f'precision@{threshold}', value=precision)
    metric_set.add(name=f'recall@{threshold}', value=recall)
    metric_set.add(name='area_under_roc', value=auroc)
    metric_set.add(name='pr_curve', value=pr_curve)
    metric_set.add(name='average_precision', value=average_precision)
    metric_set.add(name='roc_curve', value=roc)
    metric_set.add(name='calibration_curve', value=calibration_plot)
    metric_set.add(name='brier_score', value=brier)
    metric_set.add(name='predictions_histogram', value=pred_dist)

    ## View the outputs in the Metrics tab below:
    return metric_set


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7816c918-66d5-4857-b138-dfd687316487"),
    COHORT_CBC_PANEL_CURATED=Input(rid="ri.foundry.main.dataset.1dec59e8-d4ce-473e-88db-8e4d5882b73e")
)
from pyspark.sql.functions import col, create_map, lit,when
from itertools import chain

def COHORT_CBC_VALUES_PIVOT(COHORT_CBC_PANEL_CURATED):
    
    ## Spark dataframe for the comprehensive metabolic panel:
    df = COHORT_CBC_PANEL_CURATED

    ## Renaming the concept names to map to columns for pivoting the table:
    columns = {
        'Hemoglobin [Mass/volume] in Blood': 'hgb',
        'Erythrocytes [#/volume] in Blood by Automated count': 'rbc_count',
        'Leukocytes [#/volume] in Blood by Automated count': 'wbc_count',
        'Platelets [#/volume] in Blood by Automated count': 'platelet_count',
        'Hematocrit [Volume Fraction] of Blood by Automated count': 'hct',
        'MCHC [Mass/volume] by Automated count': 'mchc',
        'MCH [Entitic mass] by Automated count': 'mch',
        'MCV [Entitic volume] by Automated count': 'mcv',
        'Platelet mean volume [Entitic volume] in Blood by Automated count': 'mpv',
        'Erythrocyte distribution width [Entitic volume] by Automated count': 'rdw_volume',
        'Erythrocyte distribution width [Ratio] by Automated count': 'rdw_ratio',
        'Platelet distribution width [Entitic volume] in Blood by Automated count': 'pdw_volume',
        'Neutrophils [#/volume] in Blood by Automated count': 'absolute_neutrophils',
        'Neutrophils/100 leukocytes in Blood by Automated count': 'percent_neutrophils',
        'Lymphocytes [#/volume] in Blood by Automated count': 'absolute_lymph',
        'Lymphocytes/100 leukocytes in Blood by Automated count': 'percent_lymph',
        'Eosinophils [#/volume] in Blood by Automated count': 'absolute_eosinophils',
        'Eosinophils/100 leukocytes in Blood by Automated count': 'percent_eosinophils',
        'Basophils [#/volume] in Blood by Automated count': 'absolute_basophils', 
        'Basophils/100 leukocytes in Blood by Automated count': 'percent_basophils',
        'Monocytes [#/volume] in Blood by Automated count': 'absolute_monocytes',
        'Monocytes/100 leukocytes in Blood by Automated count': 'percent_monocytes',
        'Granulocytes/100 leukocytes in Blood by Automated count': 'percent_granulocytes',
        'Other cells [#/volume] in Blood by Automated count': 'absolute_other',
        'Other cells/100 leukocytes in Blood by Automated count': 'percent_other',
        'Variant lymphocytes [#/volume] in Blood by Automated count': 'absolute_var_lymph',
        'Variant lymphocytes/100 leukocytes in Blood by Automated count': 'percent_var_lymph',
        'Band form neutrophils/100 leukocytes in Blood by Automated count': 'absolute_band_neutrophils', 
        'Band form neutrophils [#/volume] in Blood by Automated count': 'percent_band_neutrophils',
        'CBC panel - Blood by Automated count': 'cbc_panel'
    }

    df2 = df.replace(to_replace=columns, subset=['measurement_concept_name'])

    ## Pivoting Table:
    pivotDF = df2.groupBy(["person_id", "measurement_date"]) \
                    .pivot("measurement_concept_name") \
                    .agg( F.first("value_as_number").alias("value"), F.first("unit_concept_name").alias("unit") )

## Standardising Units: 
    pivotDF = pivotDF.na.fill('gram per deciliter', subset=['hgb_unit','mchc_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'gram per deciliter', subset=['hgb_unit','mchc_unit'])
    pivotDF = pivotDF.replace('percent', 'gram per deciliter', subset=['mchc_unit'])

    #remove units other than grams per deciliter
    pivotDF = pivotDF.withColumn('hgb_value',when(col('hgb_unit')!= "gram per deciliter",None) \
    .otherwise(col('hgb_value')))
    pivotDF = pivotDF.withColumn('mchc_value',when(col('mchc_unit')!= "gram per deciliter",None) \
    .otherwise(col('mchc_value')))

    pivotDF = pivotDF.na.fill('femtoliter', subset=['mcv_unit','mpv_unit','rdw_volume_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'femtoliter', subset=['mcv_unit','mpv_unit','rdw_volume_unit'])
    #remove units other than femoliter
    pivotDF = pivotDF.withColumn('mcv_value',when(col('mcv_unit')!= "femtoliter",None) \
    .otherwise(col('mcv_value')))
    pivotDF = pivotDF.withColumn('mpv_value',when(col('mpv_unit')!= "femtoliter",None) \
    .otherwise(col('mpv_value')))
    pivotDF = pivotDF.withColumn('rdw_volume_value',when(col('rdw_volume_unit')!= "femtoliter",None) \
    .otherwise(col('rdw_volume_value')))

    pivotDF = pivotDF.na.fill('million per microliter', subset=['rbc_count_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'million per microliter', subset=['rbc_count_unit'])
    pivotDF = pivotDF.replace('trillion per liter', 'million per microliter', subset=['rbc_count_unit'])
    #remove units other than million per microliter
    pivotDF = pivotDF.withColumn('rbc_count_value',when(col('rbc_count_unit')!= "million per microliter",None) \
    .otherwise(col('rbc_count_value')))

    pivotDF = pivotDF.na.fill('picogram', subset=['mch_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'picogram', subset=['mch_unit'])
    #remove units other than million per picogram
    pivotDF = pivotDF.withColumn('mch_value',when(col('mch_unit')!= "picogram",None) \
    .otherwise(col('mch_value')))

    pivotDF = pivotDF.na.fill('thousand per microliter', subset=['absolute_basophils_unit','platelet_count_unit','absolute_neutrophils_unit','absolute_eosinophils_unit','wbc_count_unit','absolute_monocytes_unit','absolute_lymph_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'thousand per microliter', subset=['absolute_basophils_unit','platelet_count_unit','absolute_neutrophils_unit','absolute_eosinophils_unit','wbc_count_unit','absolute_monocytes_unit','absolute_lymph_unit'])
    pivotDF = pivotDF.replace('billion per liter', 'thousand per microliter', subset=['absolute_basophils_unit','platelet_count_unit','absolute_neutrophils_unit','absolute_eosinophils_unit','wbc_count_unit','absolute_monocytes_unit','absolute_lymph_unit'])
    #remove units other than thousand per microliter
    pivotDF = pivotDF.withColumn('absolute_basophils_value',when(col('absolute_basophils_unit')!= "thousand per microliter",None) \
    .otherwise(col('absolute_basophils_value')))
    pivotDF = pivotDF.withColumn('platelet_count_value',when(col('platelet_count_unit')!= "thousand per microliter",None) \
    .otherwise(col('platelet_count_value')))
    pivotDF = pivotDF.withColumn('absolute_neutrophils_value',when(col('absolute_neutrophils_unit')!= "thousand per microliter",None) \
    .otherwise(col('absolute_neutrophils_value')))
    pivotDF = pivotDF.withColumn('absolute_eosinophils_value',when(col('absolute_eosinophils_unit')!= "thousand per microliter",None) \
    .otherwise(col('absolute_eosinophils_value')))
    pivotDF = pivotDF.withColumn('wbc_count_value',when(col('wbc_count_unit')!= "thousand per microliter",None) \
    .otherwise(col('wbc_count_value')))
    pivotDF = pivotDF.withColumn('absolute_monocytes_value',when(col('absolute_monocytes_unit')!= "thousand per microliter",None) \
    .otherwise(col('absolute_monocytes_value')))
    pivotDF = pivotDF.withColumn('absolute_lymph_value',when(col('absolute_lymph_unit')!= "thousand per microliter",None) \
    .otherwise(col('absolute_lymph_value')))

    pivotDF = pivotDF.na.fill('percent', subset=['rdw_ratio_unit','hct_unit','percent_neutrophils_unit','percent_lymph_unit','percent_basophils_unit','percent_eosinophils_unit','percent_monocytes_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'percent', subset=['rdw_ratio_unit','hct_unit','percent_neutrophils_unit','percent_lymph_unit','percent_basophils_unit','percent_eosinophils_unit','percent_monocytes_unit'])
    #remove units other than percent
    pivotDF = pivotDF.withColumn('rdw_ratio_value',when(col('rdw_ratio_unit')!= "percent",None) \
    .otherwise(col('rdw_ratio_value')))
    pivotDF = pivotDF.withColumn('hct_value',when(col('hct_unit')!= "percent",None) \
    .otherwise(col('hct_value')))
    pivotDF = pivotDF.withColumn('percent_neutrophils_value',when(col('percent_neutrophils_unit')!= "percent",None) \
    .otherwise(col('percent_neutrophils_value')))
    pivotDF = pivotDF.withColumn('percent_lymph_value',when(col('percent_lymph_unit')!= "percent",None) \
    .otherwise(col('percent_lymph_value')))
    pivotDF = pivotDF.withColumn('percent_basophils_value',when(col('percent_basophils_unit')!= "percent",None) \
    .otherwise(col('percent_basophils_value')))
    pivotDF = pivotDF.withColumn('percent_eosinophils_value',when(col('percent_eosinophils_unit')!= "percent",None) \
    .otherwise(col('percent_eosinophils_value')))
    pivotDF = pivotDF.withColumn('percent_monocytes_value',when(col('percent_monocytes_unit')!= "percent",None) \
    .otherwise(col('percent_monocytes_value')))

    
    return pivotDF
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2d864a67-e164-4a23-8d26-1f36b7735eac"),
    COHORT_CMP_CURATED=Input(rid="ri.foundry.main.dataset.0b172844-9b39-499f-b71e-f026e35ce69d")
)
from pyspark.sql.functions import col, create_map, lit, when
from itertools import chain

def COHORT_CMP_VALUES_PIVOT(COHORT_CMP_CURATED):

    ## Spark dataframe for the comprehensive metabolic panel:
    df = COHORT_CMP_CURATED

    ## Renaming the concept names to map to columns for pivoting the table (Standardising):
    columns = {
        'Calcium [Mass/volume] in Serum or Plasma': 'calcium',
        'Potassium [Moles/volume] in Serum or Plasma': 'potassium',
        'Bicarbonate [Moles/volume] in Serum or Plasma': 'bicarbonate',
        'Chloride [Moles/volume] in Serum or Plasma': 'chloride', 
        'Sodium [Moles/volume] in Serum or Plasma': 'sodium',
        'Carbon dioxide, total [Moles/volume] in Serum or Plasma': 'carbon_dioxide_total',
        'Protein [Mass/volume] in Serum or Plasma': 'protein',
        'Glucose [Mass/volume] in Serum or Plasma': 'glucose',
        'Creatinine [Mass/volume] in Serum or Plasma': 'creatinine',
        'Bilirubin.total [Mass/volume] in Serum or Plasma': 'bilirubin_total',
        'Globulin [Mass/volume] in Serum by calculation': 'globulin',
        'Albumin/Globulin [Mass Ratio] in Serum or Plasma': 'albumin_globulin_ratio', 
        'Albumin [Mass/volume] in Serum or Plasma by Bromocresol green (BCG) dye binding method': 'albumin_bcg', 
        'Albumin [Mass/volume] in Serum or Plasma': 'albumin',
        'Albumin [Mass/volume] in Serum or Plasma by Electrophoresis': 'albumin_electrophoresis',
        'Albumin [Mass/volume] in Serum or Plasma by Bromocresol purple (BCP) dye binding method': 'albumin_bcp',
        'Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma': 'alkaline_phosphatase',
        'Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma': 'alanine_aminotransferase',
        'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma': 'aspartate_aminotransferase',
        'Anion gap in Serum or Plasma': 'anion_gap',
        'Urea nitrogen [Mass/volume] in Serum or Plasma': 'bun',
        'Urea nitrogen/Creatinine [Mass Ratio] in Serum or Plasma': 'bun_creatinine_ratio',  
        'Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)': 'glomerular_filt_blacks_MDRD',
        'Glomerular filtration rate/1.73 sq M.predicted among non-blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)': 'glomerular_filt_nonblacks_MDRD',
        'Glomerular filtration rate/1.73 sq M.predicted among females [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)': 'glomerular_filt_females_MDRD',
        'Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)': 'glomerular_filt_blacks_CKD',         
        'Glomerular filtration rate/1.73 sq M.predicted among non-blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)': 'glomerular_filt_nonblacks_CKD',
        'Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)': 'glomerular_filt_CKD',
    }
    df2 = df.replace(to_replace=columns, subset=['measurement_concept_name'])

    ## Pivoting Table:
    pivotDF = df2.groupBy(["person_id", "measurement_date"]) \
                    .pivot("measurement_concept_name") \
                    .agg( F.first("value_as_number").alias("value"), F.first("unit_concept_name").alias("unit") )

    ## Standardising Units: 
    pivotDF = pivotDF.na.fill('millimole per liter', subset=['sodium_unit','chloride_unit','potassium_unit','carbon_dioxide_total_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'millimole per liter', subset=['sodium_unit','chloride_unit','potassium_unit','carbon_dioxide_total_unit'])
    #remove units other than millimole per liter
    pivotDF = pivotDF.withColumn('sodium_value',when(col('sodium_unit')!= "millimole per liter",None) \
    .otherwise(col('sodium_value')))
    pivotDF = pivotDF.withColumn('chloride_value',when(col('chloride_unit')!= "millimole per liter",None) \
    .otherwise(col('chloride_value')))
    pivotDF = pivotDF.withColumn('potassium_value',when(col('potassium_unit')!= "millimole per liter",None) \
    .otherwise(col('potassium_value')))
    pivotDF = pivotDF.withColumn('carbon_dioxide_total_value',when(col('carbon_dioxide_total_unit')!= "millimole per liter",None) \
    .otherwise(col('carbon_dioxide_total_value')))

    pivotDF = pivotDF.na.fill('milligram per deciliter', subset=['bun_unit', 'calcium_unit','creatinine_unit','glucose_unit','bilirubin_total_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'milligram per deciliter', subset=['bun_unit', 'calcium_unit', 'creatinine_unit','glucose_unit','bilirubin_total_unit'])
    #remove units other than milligram per deciliter
    pivotDF = pivotDF.withColumn('bun_value',when(col('bun_unit')!= "milligram per deciliter",None) \
    .otherwise(col('bun_value')))
    pivotDF = pivotDF.withColumn('calcium_value',when(col('calcium_unit')!= "milligram per deciliter",None) \
    .otherwise(col('calcium_value')))
    pivotDF = pivotDF.withColumn('creatinine_value',when(col('creatinine_unit')!= "milligram per deciliter",None) \
    .otherwise(col('creatinine_value')))
    pivotDF = pivotDF.withColumn('glucose_value',when(col('glucose_unit')!= "milligram per deciliter",None) \
    .otherwise(col('glucose_value')))
    pivotDF = pivotDF.withColumn('bilirubin_total_value',when(col('bilirubin_total_unit')!= "milligram per deciliter",None) \
    .otherwise(col('bilirubin_total_value')))

    pivotDF = pivotDF.na.fill('unit per liter', subset=['alkaline_phosphatase_unit','aspartate_aminotransferase_unit','alanine_aminotransferase_unit'])
    pivotDF = pivotDF.replace('international unit per liter', 'unit per liter', subset=['alkaline_phosphatase_unit','aspartate_aminotransferase_unit','alanine_aminotransferase_unit'])    
    pivotDF = pivotDF.replace('No matching concept','unit per liter', subset=['aspartate_aminotransferase_unit','alanine_aminotransferase_unit']) 
    #remove units other than unit per liter
    pivotDF = pivotDF.withColumn('alkaline_phosphatase_value',when(col('alkaline_phosphatase_unit')!= "unit per liter",None) \
    .otherwise(col('alkaline_phosphatase_value')))
    pivotDF = pivotDF.withColumn('aspartate_aminotransferase_value',when(col('aspartate_aminotransferase_unit')!= "unit per liter",None) \
    .otherwise(col('aspartate_aminotransferase_value')))
    pivotDF = pivotDF.withColumn('alanine_aminotransferase_value',when(col('alanine_aminotransferase_unit')!= "unit per liter",None) \
    .otherwise(col('alanine_aminotransferase_value')))

    pivotDF = pivotDF.na.fill('gram per deciliter', subset=['protein_unit','albumin_unit'])
    pivotDF = pivotDF.replace('No matching concept', 'gram per deciliter', subset=['protein_unit','albumin_unit'])
    #remove units other than milligram per deciliter
    pivotDF = pivotDF.withColumn('protein_value',when(col('protein_unit')!= "gram per deciliter",None) \
    .otherwise(col('protein_value')))
    pivotDF = pivotDF.withColumn('albumin_value',when(col('albumin_unit')!= "gram per deciliter",None) \
    .otherwise(col('albumin_value')))

    return pivotDF
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9aae6f0d-e262-4c7f-ba7c-795814f0787b"),
    COHORT_IMMUNOASSAY_CURATED=Input(rid="ri.foundry.main.dataset.cd8394ac-1a63-41f3-a52c-eef5c28ffe57")
)
def COHORT_IMMUNOGLOBIN_PIVOT( COHORT_IMMUNOASSAY_CURATED):
    ## Spark dataframe for the comprehensive metabolic panel:
    df = COHORT_IMMUNOASSAY_CURATED

    ## Renaming the concept names to map to columns for pivoting the table:
    columns = {
        'SARS-CoV-2 (COVID-19) Ab [Presence] in Serum or Plasma by Immunoassay': 'cov2_Ab_presence',
        'SARS-CoV-2 (COVID-19) Ab panel - Serum or Plasma by Immunoassay': 'cov2_Ab_panel',
        'SARS-CoV-2 (COVID-19) Ab [Units/volume] in Serum or Plasma by Immunoassay': 'cov2_Ab_amt',
        'SARS-CoV-2 (COVID-19) Ag [Presence] in Respiratory specimen by Rapid immunoassay': 'cov2_Ag_presence_rapid',
        'SARS-CoV-2 (COVID-19) Ag [Presence] in Upper respiratory specimen by Immunoassay': 'cov2_Ag_presence', 
        'SARS-CoV-2 (COVID-19) Ag [Presence] in Upper respiratory specimen by Rapid immunoassay': 'cov2_Ag_presence_rapid',
        'SARS-CoV+SARS-CoV-2 (COVID-19) Ag [Presence] in Respiratory specimen by Rapid immunoassay': 'cov2_Ag_presence',  
        'SARS-CoV-2 (COVID-19) IgG Ab [Presence] in Serum, Plasma or Blood by Rapid immunoassay': 'cov2_IgG_Ab_presence_rapid',
        'SARS-CoV-2 (COVID-19) IgG Ab [Presence] in DBS by Immunoassay': 'cov2_IgG_Ab_presence_DBS',
        'SARS-CoV-2 (COVID-19) IgG Ab [Presence] in Serum or Plasma by Immunoassay': 'cov2_IgG_Ab_presence',
        'SARS-CoV-2 (COVID-19) IgG Ab [Units/volume] in Serum or Plasma by Immunoassay': 'cov2_IgG_Ab_amt',
        'SARS-CoV-2 (COVID-19) IgA Ab [Presence] in Serum or Plasma by Immunoassay': 'cov2_IgA_Ab_presence', 
        'Influenza virus A and B and SARS-CoV+SARS-CoV-2 (COVID-19) Ag panel - Upper respiratory specimen by Rapid immunoassay': 'influ_cov2_Ag_panel', 
        'Influenza virus A Ag [Presence] in Upper respiratory specimen by Rapid immunoassay': 'influA_Ag_presence',
        'Influenza virus B Ag [Presence] in Upper respiratory specimen by Rapid immunoassay': 'influB_Ag_presence', 
        'SARS-CoV-2 (COVID-19) IgM Ab [Units/volume] in Serum or Plasma by Immunoassay': 'cov2_IgM_Ab_amt',
        'SARS-CoV-2 (COVID-19) IgM Ab [Presence] in Serum, Plasma or Blood by Rapid immunoassay': 'cov2_IgA_Ab_presence_rapid', 
        'SARS-CoV-2 (COVID-19) IgM Ab [Presence] in Serum or Plasma by Immunoassay': 'cov2_IgA_Ab_presence', 
        'SARS-CoV-2 (COVID-19) IgG+IgM Ab [Presence] in Serum or Plasma by Immunoassay': 'cov2_IgG_IgM_Ab_presence'
    }

    df2 = df.replace(to_replace=columns, subset=['measurement_concept_name'])

    ## Pivoting Table:
    pivotDF = df2.groupBy(["person_id", "measurement_date"]) \
                    .pivot("measurement_concept_name") \
                    .agg( F.first("value_as_number").alias("value") ) \
                    .orderBy(["person_id", "measurement_date"])

    return pivotDF
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.489b4135-2d60-45c7-ac58-b2d6b161e4b3"),
    COHORT_WT_HT_CURATED=Input(rid="ri.foundry.main.dataset.2eeb4595-97b9-45a3-9430-6b76e738a3e0")
)
from pyspark.sql.functions import col, create_map, lit
from itertools import chain

def COHORT_WT_HT_PIVOT(COHORT_WT_HT_CURATED):
    ## Spark dataframe for the comprehensive metabolic panel:
    df = COHORT_WT_HT_CURATED

    ## Renaming the concept names to map to columns for pivoting the table:
    columns = {
        'Body weight': 'body_weight', 
        'Body weight Stated': 'body_weight', 
        'Body weight - Reported --usual': 'body_weight', 
        'Body weight Measured': 'body_weight',
        'Body weight Measured --ante partum':'body_weight_msc', 
        'Dry body weight Estimated':'body_weight_msc', 
        'Body weight Measured --pre pregnancy': 'body_weight_msc',
        'Dry body weight Measured':'dry_body_height',
        'Body weight Measured --when specimen taken':'body_weight_msc', 
        'Body weight --used for drug calculation':'body_weight_msc',
        'Body height Measured':'body_height', 
        'Body height Stated': 'body_height', 
        'Body height measure': 'body_height', 
        'Body height':'body_height', 
        'Body height --standing':'body_height'
    }

    df2 = df.replace(to_replace=columns, subset=['measurement_concept_name'])

    ## Pivoting Table:
    pivotDF = df2.groupBy(["person_id", "measurement_date"]) \
                    .pivot("measurement_concept_name") \
                    .agg( F.first("value_as_number").alias("value"), F.first("unit_concept_name").alias("unit"))
    

    return pivotDF
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5f17bb35-d2dc-44e5-af86-9dff02819bba"),
    COHORT_DIAGNOSIS_CURATED=Input(rid="ri.foundry.main.dataset.232c95d8-1afb-4f60-bfbf-3c1bb7f15606")
)
#from pyspark.sql.functions import expr

def Cohort_dx_ct_features(COHORT_DIAGNOSIS_CURATED):
    ## Initialize the input with the table:
    df = COHORT_DIAGNOSIS_CURATED

    ## Rename measurement concept name for future use: 
    name_dict = {
        "DIABETES COMPLICATED": "diabetes_complicated",
        "DIABETES UNCOMPLICATED": "diabetes_uncomplicated",
        "[VSAC] Asthma ": "asthma",
        "[L3C][GAIL] COPD": "copd",
        'HYPERTENSION': 'hypertension', 
        'HEART FAILURE': 'heart_failure',  
        'OBESITY': 'obesity', 
        'TOBACCO SMOKER': 'tobacco_smoker'
    } 
    df = df.replace(to_replace=name_dict, subset=['concept_set_name'])

    ## Pivoting the table and gathering the related diagnosis count:
    pivotDF = df.groupBy(["person_id"]).pivot("concept_set_name").agg(F.count("concept_set_name").alias("count")).na.fill(0)
    
    return pivotDF

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a238f428-a89e-44c2-9e69-1c8e6cb4c367"),
    missing_value_imputation_test=Input(rid="ri.foundry.main.dataset.4192cd64-6a5b-40c8-96d0-c95a4c570c3a"),
    xgb_train_test_model=Input(rid="ri.foundry.main.dataset.6d6663d0-454f-426c-96c3-dea9f75da0e3")
)
def GAIL_predictions(xgb_train_test_model, missing_value_imputation_test):
    
    ## Initializing the model: 
    xgb_benchmark_model = xgb_train_test_model

    ## Initializing the data on which the predictions are to be made: 
    df_test = missing_value_imputation_test

    # Initializing the decision threshold: 
    threshold = 0.5

    # Generate the prediction probabilities for the extracted features:
    df_pred = xgb_benchmark_model.transform( df_test[features] ) # prediction probabilities
    df_pred['person_id'] = df_test['person_id']

    # Generating scores as per needed: 
    scores = df_pred[['person_id', 'probability_1']].rename( columns={'probability_1': 'outcome_likelihood'} )

    return scores
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.30aed68b-b11d-40a0-879a-fc5d14b48878"),
    COHORT_WT_HT_PIVOT=Input(rid="ri.foundry.main.dataset.489b4135-2d60-45c7-ac58-b2d6b161e4b3")
)
from pyspark.sql.functions import col, create_map, lit,when
from itertools import chain

def HT_WT_standard(COHORT_WT_HT_PIVOT):
    ## Spark dataframe for HT WT:
    df = COHORT_WT_HT_PIVOT

    df3 = df.withColumn('body_height_value', when(col('body_height_unit')== "inch (US)",col('body_height_value')*2.54) \
        .when(col('body_height_unit')== "inch (international)",col('body_height_value')*2.54) \
        .when(col('body_height_unit')== "Inches",col('body_height_value')*2.54) \
        .when(col('body_height_unit')== "meter",col('body_height_value')*100) \
        .when(col('body_height_unit').isNull(),None) \
        .when(col('body_height_unit')== "No matching concept",None) \
        .when(col('body_height_unit')== "percent",None) \
        .when(col('body_height_unit')== "milliliter",None) \
        .otherwise(col('body_height_value')))
    
    df3 = df3.withColumn('body_height_unit',when(col('body_height_unit')== "inch (US)",'centimeter') \
        .when(col('body_height_unit')== "inch (international)",'centimeter') \
        .when(col('body_height_unit')== "Inches",'centimeter') \
        .when(col('body_height_unit')== "meter",'centimeter') \
        .when(col('body_height_unit')== "No matching concept",None) \
        .when(col('body_height_unit')== "percent",None) \
        .when(col('body_height_unit')== "milliliter",None) \
        .otherwise(col('body_height_unit')))

    df3 = df3.withColumn('body_weight_value',when(col('body_weight_unit')== "kilogram",col('body_weight_value')*2.20462) \
        .when(col('body_weight_unit')== "ounce (avoirdupois)",col('body_weight_value')*0.0625) \
        .when(col('body_weight_unit')== "gram",col('body_weight_value')*0.00220462) \
        .when(col('body_weight_unit')== "fluid ounce (US)",None) \
        .when(col('body_weight_unit')== "oz",None) \
        .when(col('body_weight_unit')== "No matching concept",None) \
        .when(col('body_weight_unit')== "percent",None) \
        .when(col('body_weight_unit')== "meter",None) \
        .when(col('body_weight_unit')== "milliliter",None) \
        .when(col('body_weight_unit').isNull(),None) \
        .otherwise(col('body_weight_value')))

    df3 = df3.withColumn('body_weight_unit',when(col('body_weight_unit')== "kilogram",'pound (US)') \
        .when(col('body_weight_unit')== "ounce (avoirdupois)",'pound (US)') \
        .when(col('body_weight_unit')== "gram",'pound (US)') \
        .when(col('body_weight_unit')== "fluid ounce (US)",None) \
        .when(col('body_weight_unit')== "oz",None) \
        .when(col('body_weight_unit')== "No matching concept",None) \
        .when(col('body_weight_unit')== "percent",None) \
        .when(col('body_weight_unit')== "meter",None) \
        .when(col('body_weight_unit')== "milliliter",None) \
        .otherwise(col('body_weight_unit')))

    df3 = df3.drop('body_weight_msc_value','body_weight_msc_unit','dry_body_height_unit','dry_body_height_value')

    
    
    return df3

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.51d16276-097f-4867-8e96-297bae95952e"),
    measurements_standard_features_final=Input(rid="ri.foundry.main.dataset.9432e2d7-7f99-4069-80d9-af883df8efa3")
)
def Measurements_after(measurements_standard_features_final):

    df = measurements_standard_features_final
    features = df.columns
    features.remove('status_relative_index')

    return df.filter(df.status_relative_index == "after").select(*features)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.21288a64-d469-495e-ae9b-29d83d29a352"),
    measurements_standard_features_final=Input(rid="ri.foundry.main.dataset.9432e2d7-7f99-4069-80d9-af883df8efa3")
)
def Measurements_during(measurements_standard_features_final):

    df = measurements_standard_features_final
    features = df.columns
    features.remove('status_relative_index')

    return df.filter(df.status_relative_index == "during").select(*features)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.edc9df8d-43c1-4fb4-a893-db611e8b61ea"),
    measurements_standard_features_final=Input(rid="ri.foundry.main.dataset.9432e2d7-7f99-4069-80d9-af883df8efa3")
)
def Measurements_features_before(measurements_standard_features_final):

    df = measurements_standard_features_final
    features = df.columns
    features.remove('status_relative_index')

    return df.filter(df.status_relative_index == "before").select(*features)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d2c55b64-b2ff-485c-beac-e5b5ac108940"),
    meds_grouped=Input(rid="ri.foundry.main.dataset.a9c6a2e1-4269-4b30-b475-4160cf9007af")
)
def Meds_dataset_v1(meds_grouped):
    
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

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a2f26681-acd9-4fca-9a65-73a9562c7697"),
    procedure_time_pivoted=Input(rid="ri.foundry.main.dataset.56726a0b-fd5f-40e4-b017-bf497bc12c0a")
)
def Procedure_timeindex_dataset_V1( procedure_time_pivoted):
   df =  procedure_time_pivoted
   df1 = df.groupBy("person_id"). pivot("feature_time_index"). agg(F.max("Ventilator_used").alias('Ventilator_used'),F.max('Lungs_CT_scan').alias('Lungs_CT_scan'),F.max('Chest_X_ray').alias('Chest_X_ray'),F.max('Lung_Ultrasound').alias('Lung_Ultrasound'),F.max('ECMO_performed').alias('ECMO_performed'),F.max('ECG_performed').alias('ECG_performed'), F.max('Echocardiogram_performed').alias('Echocardiogram_performed'),F.max('Blood_transfusion').alias('Blood_transfusion')).na.fill(0)
   return df1
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.83c4a22b-0a6a-48de-a2e9-c22ae538bd9c"),
    vitals_grouped=Input(rid="ri.foundry.main.dataset.b22f0cc5-51b8-4e5d-a46a-c4dff3a0f5f8")
)
def Vitals_Dataset_V1(vitals_grouped):
    # Pivot on feature_name
    vitals_pivoted = vitals_grouped.groupBy(["person_id", "before_or_after_index"]).pivot("feature_name").agg(F.first("vital_avg").alias("vital_avg"))#, F.first("vital_stddev").alias("vital_stddev"))
    
    # Break into 3 dfs (before, during, after)
    vitals_pivoted_before = vitals_pivoted.filter(vitals_pivoted.before_or_after_index == "before") 
    vitals_pivoted_during = vitals_pivoted.filter(vitals_pivoted.before_or_after_index == "during")
    vitals_pivoted_after  = vitals_pivoted.filter(vitals_pivoted.before_or_after_index == "after")

    # Change column names to add prefix using global function rename_cols()
    vitals_pivoted_before = rename_cols(vitals_pivoted_before, suffix="_before")
    vitals_pivoted_during = rename_cols(vitals_pivoted_during, suffix="_during")
    vitals_pivoted_after  = rename_cols(vitals_pivoted_after,  suffix="_after")

    # Outer join the 3 together on person_id
    vitals_df = vitals_pivoted_before\
                    .join(vitals_pivoted_during, on=("person_id"), how="outer")\
                    .join(vitals_pivoted_after,  on=("person_id"), how="outer")

    return vitals_df
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.1ba45b78-bb25-4ba7-9baf-755acd57d460"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence_merge=Input(rid="ri.foundry.main.dataset.ed6feda8-c72f-4a3e-8c6e-a97c6695d0f2"),
    microvisits_to_macrovisits_merge=Input(rid="ri.foundry.main.dataset.fa1a8bfa-4e83-4fb8-9928-13378f987820"),
    observation_merge=Input(rid="ri.foundry.main.dataset.67d6b203-2db2-4289-a12b-027fe2b52f22"),
    procedure_occurrence_merge=Input(rid="ri.foundry.main.dataset.44ffaa66-95d6-4323-817d-4bbf1bf84f12")
)
# Add_ICU (d5691458-1e67-4887-a68f-4cfbd4753295): v1

def add_icu(microvisits_to_macrovisits_merge, concept_set_members, procedure_occurrence_merge, condition_occurrence_merge, observation_merge):
    icu_codeset_id = 469361388

    icu_concepts = concept_set_members.filter(F.col('codeset_id') == icu_codeset_id).select('concept_id','concept_name')

    procedures_df = procedure_occurrence_merge[['visit_occurrence_id', 'procedure_concept_id']]
    condition_df = condition_occurrence_merge[['visit_occurrence_id','condition_concept_id']]
    observation_df = observation_merge[['visit_occurrence_id','observation_concept_id']]

    df = microvisits_to_macrovisits_merge

    df = with_concept_name(df, procedures_df, icu_concepts, 'procedure_concept_id', 'procedure_concept_name')
    df = with_concept_name(df, condition_df, icu_concepts, 'condition_concept_id', 'condition_concept_name')
    df = with_concept_name(df, observation_df, icu_concepts, 'observation_concept_id', 'observation_concept_name')

    df = df.withColumn('is_icu', F.when(F.coalesce(df['procedure_concept_name'], df['condition_concept_name'], df['observation_concept_name']).isNotNull(), 1).otherwise(0))

    return df

    
    

#################################################
## Global imports and functions included below ##
#################################################

from pyspark.sql import functions as F
from pyspark.sql.window import Window

COERCE_LOS_OUTLIERS = False
LOS_MAX = 60

def with_concept_name(df, domain_df, icu_concepts, concept_id_column, concept_name_column):

    df = df.join(domain_df, 'visit_occurrence_id', how='left')

    df = df.join(icu_concepts, df[concept_id_column] == icu_concepts['concept_id'], how='left')

    df = df.withColumnRenamed("concept_name", concept_name_column).drop('concept_id').distinct()

    df = df.withColumn('rn', F.row_number().over(Window.partitionBy('visit_occurrence_id').orderBy(F.desc(concept_name_column))))

    df = df.filter(df['rn'] == 1)

    df = df.drop('rn')

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bb137691-7bd6-4df6-adf6-71efebaea2c5"),
    add_ed_ip_op_copied=Input(rid="ri.foundry.main.dataset.c82ed648-a060-4559-87b8-30d3d03065a9"),
    index_range=Input(rid="ri.foundry.main.dataset.37168eef-4434-4de4-9fca-2d1ab449e3b2")
)
def after_index_visit_name_counts_copied(add_ed_ip_op_copied, index_range):
    idx_df = index_range.select('person_id', 'index_start_date', 'index_end_date')
    add_ed_ip_op = add_ed_ip_op_copied
    df = add_ed_ip_op.join(idx_df, 'person_id', how='left')
    during_df = df.where(F.col('visit_start_date') > F.col('index_end_date'))
    
    counts_df = during_df.groupBy('person_id').agg(
        F.sum("is_ed").alias("after_ed_cnt"),
        F.sum("is_ip").alias("after_ip_cnt"),
        F.sum("is_op").alias("after_op_cnt"),
        F.sum("is_tele").alias("after_tele_cnt"),
        )
    
    return counts_df
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9b29b83d-2db0-4e05-99cd-ac844d026435"),
    add_ed_ip_op_copied=Input(rid="ri.foundry.main.dataset.c82ed648-a060-4559-87b8-30d3d03065a9"),
    index_range=Input(rid="ri.foundry.main.dataset.37168eef-4434-4de4-9fca-2d1ab449e3b2")
)
def before_index_visit_name_counts_copied(add_ed_ip_op_copied, index_range):
    idx_df = index_range.select('person_id', 'index_start_date', 'index_end_date')
    add_ed_ip_op = add_ed_ip_op_copied
    df = add_ed_ip_op.join(idx_df, 'person_id', how='left')
    before_df = df.where(F.coalesce(F.col('visit_end_date'), F.col('visit_start_date')) < F.col('index_start_date'))
    
    counts_df = before_df.groupBy('person_id').agg(
        F.sum("is_ed").alias("before_ed_cnt"),
        F.sum("is_ip").alias("before_ip_cnt"),
        F.sum("is_op").alias("before_op_cnt"),
        F.sum("is_tele").alias("before_tele_cnt"))
    
    return counts_df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9605effb-6651-49ba-969d-8c79bda5bce6"),
    add_los_and_index=Input(rid="ri.vector.main.execute.a8e8ce77-0c4b-42c2-92b9-c30c3f90d14b")
)
def coerce_los_outliers(add_los_and_index):
    df = add_los_and_index
    
    if COERCE_LOS_OUTLIERS:
        df = df.withColumn('los_mod', F.when(F.col('los') > LOS_MAX, 0).when(F.col('los') < 0, 0).otherwise(F.col('los')))
        df = df.drop('los').withColumnRenamed('los_mod','los')

    return df

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0b095fdd-ee5e-4864-9ee7-8a351c4708af"),
    add_ed_ip_op_copied=Input(rid="ri.foundry.main.dataset.c82ed648-a060-4559-87b8-30d3d03065a9"),
    index_range=Input(rid="ri.foundry.main.dataset.37168eef-4434-4de4-9fca-2d1ab449e3b2")
)
def during_index_visit_name_counts_copied(add_ed_ip_op_copied, index_range):
    idx_df = index_range.select('person_id', 'index_start_date', 'index_end_date')
    add_ed_ip_op = add_ed_ip_op_copied
    df = add_ed_ip_op.join(idx_df, 'person_id', how='left')
    
    during_df = df.where((F.col('visit_start_date') >= F.col('index_start_date')) & (F.coalesce(F.col('visit_end_date'), F.col('visit_start_date')) <= F.col('index_end_date')))
    
    counts_df = during_df.groupBy('person_id').agg(F.sum("is_ed").alias("during_ed_cnt"),F.sum("is_ip").alias("during_ip_cnt"),F.sum("is_op").alias("during_op_cnt"),F.sum("is_tele").alias("during_tele_cnt"))
    
    return counts_df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e913f8c5-a8ba-4297-a777-add026a683b1"),
    get_training_dataset=Input(rid="ri.foundry.main.dataset.3c0351f5-f2e3-4939-bd52-c106b7c0752f"),
    xgb_train_test_model=Input(rid="ri.foundry.main.dataset.6d6663d0-454f-426c-96c3-dea9f75da0e3")
)
def get_predictions(xgb_train_test_model, get_training_dataset):
    xgb_benchmark_model = xgb_train_test_model

    ## Initializing the data on which the predictions are to be made: 
    ## Subsetting the imported dataset to include the desired features (listed in 'features') and the y_true described by 'label'
    # features and label are initialied in the global code 
    df_train = get_training_dataset[features+[label]]

    # Initializing the decision threshold: 
    threshold = 0.5

    # Generate the prediction probabilities for the extracted features:
    df_pred = xgb_benchmark_model.transform(df_train[features]) # prediction probabilities
    df_pred['true_label'] = df_train[label] # Appending the true labels
    df_pred['pred_label'] = (df_pred.probability_1 > threshold).astype('int')

    return df_pred
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3cd10bcd-fead-4854-8029-b1005944e679"),
    get_training_dataset=Input(rid="ri.foundry.main.dataset.3c0351f5-f2e3-4939-bd52-c106b7c0752f"),
    missing_value_imputation=Input(rid="ri.foundry.main.dataset.3bc6016d-2f9c-464b-8c4e-296095919e7b")
)
def get_testing_dataset(get_training_dataset, missing_value_imputation):
    
    df_feat = missing_value_imputation
    df_train = get_training_dataset

    ##df_test = df_feat.filter( ~F.col('person_id').isin( df_train.person_id.collect() ) )

    return df_feat.join(df_train, df_feat.person_id==df_train.person_id, "left_anti")
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3c0351f5-f2e3-4939-bd52-c106b7c0752f"),
    missing_value_imputation=Input(rid="ri.foundry.main.dataset.3bc6016d-2f9c-464b-8c4e-296095919e7b")
)
def get_training_dataset(missing_value_imputation):
    
    ## Feature Dataset Initialization:
    df_feat = missing_value_imputation

    ## Training and testing split: 
    (trainingData, testData) = df_feat.randomSplit([0.75, 0.25], seed=random_seed)
    
    return trainingData
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7ff39028-5426-4c6d-aece-f6e7e4e0611e"),
    COHORT_CBC_VALUES_PIVOT=Input(rid="ri.foundry.main.dataset.7816c918-66d5-4857-b138-dfd687316487"),
    COHORT_CMP_VALUES_PIVOT=Input(rid="ri.foundry.main.dataset.2d864a67-e164-4a23-8d26-1f36b7735eac"),
    HT_WT_standard=Input(rid="ri.foundry.main.dataset.30aed68b-b11d-40a0-879a-fc5d14b48878"),
    index_range=Input(rid="ri.foundry.main.dataset.37168eef-4434-4de4-9fca-2d1ab449e3b2")
)
def measurements_standard_features(HT_WT_standard, COHORT_CBC_VALUES_PIVOT, COHORT_CMP_VALUES_PIVOT, index_range):

    from re import match

    ## Initialize all measurement related features:
    df_htwt = HT_WT_standard.withColumnRenamed('person_id', 'person_id_htwt') \
                            .withColumnRenamed('measurement_date', 'measurement_date_htwt')
    df_cbc = COHORT_CBC_VALUES_PIVOT.withColumnRenamed('person_id', 'person_id_cbc') \
                                .withColumnRenamed('measurement_date', 'measurement_date_cbc') 
    df_cmp = COHORT_CMP_VALUES_PIVOT.withColumnRenamed('person_id', 'person_id_cmp') \
                                    .withColumnRenamed('measurement_date', 'measurement_date_cmp')
    df_covid = index_range.withColumnRenamed('person_id', 'person_id_covid')

    ## Joins: 
    df_feat = df_htwt.join(df_cmp, [df_htwt.person_id_htwt == df_cmp.person_id_cmp, df_htwt.measurement_date_htwt == df_cmp.measurement_date_cmp], 'outer') \
                        .join(df_cbc, [df_htwt.person_id_htwt == df_cbc.person_id_cbc, df_htwt.measurement_date_htwt == df_cbc.measurement_date_cbc], 'outer')
    
    ## generating a common person_id and measurement_date column
    df_feat = df_feat.withColumn('person_id', F.greatest(F.col('person_id_htwt'), F.col('person_id_cmp'), F.col('person_id_cbc')) )
    df_feat = df_feat.withColumn('measurement_date', F.greatest(F.col('measurement_date_htwt'), F.col('measurement_date_cmp'), F.col('measurement_date_cbc')) )
    
    ## Adding a new column for before, during, after:
    df_feat = df_feat.join( df_covid.select('person_id_covid', 'index_start_date', 'index_end_date'), df_feat.person_id == df_covid.person_id_covid, 'left')
    df_feat = df_feat.withColumn( 'status_relative_index',  when( col('measurement_date') > col('index_end_date'), 'after' ) \
                                                            .when( col('measurement_date') < col('index_start_date'), 'before' ) \
                                                            .otherwise( 'during' ) 
                                )
                                
    ## Dropping unnecessary columns:
    cols = df_feat.columns 
    filtered_cols = list(filter(lambda v: match(".*unit$", v), cols))
    df_feat = df_feat.drop('person_id_htwt', 'measurement_date_htwt', 'person_id_cbc', 'measurement_date_cbc', 'person_id_cmp', 'measurement_date_cmp', 'person_id_covid', 'index_start_date', 'index_end_date', 'measurement_date', *filtered_cols)

    return df_feat
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9432e2d7-7f99-4069-80d9-af883df8efa3"),
    measurements_standard_features=Input(rid="ri.foundry.main.dataset.7ff39028-5426-4c6d-aece-f6e7e4e0611e")
)
def measurements_standard_features_final(measurements_standard_features):

    df = measurements_standard_features
    features = df.columns
    features.remove('person_id')
    features.remove('status_relative_index')

    ## Pivoting:
    df_grouped = df.groupBy("person_id", "status_relative_index").mean()

    for name in features:
       df_grouped = df_grouped.withColumnRenamed( "avg("+name+")", name   )
 

    return df_grouped
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.16159dd7-bca0-4f6d-9d90-0f8ee980241b"),
    Cohort_dx_ct_features=Input(rid="ri.foundry.main.dataset.5f17bb35-d2dc-44e5-af86-9dff02819bba"),
    Measurements_after=Input(rid="ri.foundry.main.dataset.51d16276-097f-4867-8e96-297bae95952e"),
    Measurements_during=Input(rid="ri.foundry.main.dataset.21288a64-d469-495e-ae9b-29d83d29a352"),
    Measurements_features_before=Input(rid="ri.foundry.main.dataset.edc9df8d-43c1-4fb4-a893-db611e8b61ea"),
    Meds_dataset_v1=Input(rid="ri.foundry.main.dataset.d2c55b64-b2ff-485c-beac-e5b5ac108940"),
    Procedure_timeindex_dataset_V1=Input(rid="ri.foundry.main.dataset.a2f26681-acd9-4fca-9a65-73a9562c7697"),
    Vitals_Dataset_V1=Input(rid="ri.foundry.main.dataset.83c4a22b-0a6a-48de-a2e9-c22ae538bd9c"),
    person_demographics=Input(rid="ri.foundry.main.dataset.f4b12b19-5966-467f-b4d8-5e4a87b3f5fd"),
    smoking_dataset_1=Input(rid="ri.foundry.main.dataset.34d2fd38-d03f-4f08-8a99-bd6617c828be"),
    utilization_updated_columns=Input(rid="ri.foundry.main.dataset.b1c6e531-a1f8-4d20-b258-b947ff462747")
)
def merge_testing_data(person_demographics, Measurements_features_before, Measurements_during, Measurements_after, Vitals_Dataset_V1, smoking_dataset_1, utilization_updated_columns, Cohort_dx_ct_features, Procedure_timeindex_dataset_V1, Meds_dataset_v1):
     
    ## Including all tables for the modeling:
    df_pat = person_demographics.select("*").where("isTrainSet == 0")
        
    df_dxct = Cohort_dx_ct_features.withColumnRenamed('person_id', 'person_id_dxct')
    df_proc = Procedure_timeindex_dataset_V1.withColumnRenamed('person_id', 'person_id_proc')
    df_util = utilization_updated_columns.withColumnRenamed('person_id', 'person_id_utl')    
        
    df_meas_after = Measurements_after.select(*[col(c).alias("after_"+c) for c in Measurements_after.columns])
    df_meas_before = Measurements_features_before.select(*[col(c).alias("before_"+c) for c in Measurements_features_before.columns])
    df_meas_during = Measurements_during.select(*[col(c).alias("during_"+c) for c in Measurements_during.columns])
    df_vital = Vitals_Dataset_V1.withColumnRenamed('person_id', 'person_id_vital')
    df_meds = Meds_dataset_v1.withColumnRenamed('person_id', 'person_id_meds')
    df_smoke = smoking_dataset_1.withColumnRenamed('person_id', 'person_id_smoke')

    df_test = df_pat.join(df_dxct, df_pat.person_id == df_dxct.person_id_dxct, 'left') \
                        .join(df_proc, df_pat.person_id == df_proc.person_id_proc, 'left') \
                        .join(df_util, df_pat.person_id == df_util.person_id_utl, 'left') \
                        .join(df_meas_after, df_pat.person_id == df_meas_after.after_person_id, 'left') \
                        .join(df_meas_before, df_pat.person_id == df_meas_before.before_person_id, 'left') \
                        .join(df_meas_during, df_pat.person_id == df_meas_during.during_person_id, 'left') \
                        .join(df_vital, df_pat.person_id == df_vital.person_id_vital, 'left') \
                        .join(df_meds, df_pat.person_id == df_meds.person_id_meds, 'left') \
                        .join(df_smoke, df_pat.person_id == df_smoke.person_id_smoke, 'left') 

    ## Drop unnecessary columns:
    df_test = df_test.drop('person_id_dxct', 'person_id_proc', 'person_id_utl', 
                            'after_person_id', 'before_person_id', 'mduring_person_id', 'person_id_vital', 'person_id_meds', 'person_id_smoke', 'covid_index')

    return df_test
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bae2ad21-0aaa-4216-9208-c9ddc20b044d"),
    Cohort_dx_ct_features=Input(rid="ri.foundry.main.dataset.5f17bb35-d2dc-44e5-af86-9dff02819bba"),
    Long_COVID_Silver_Standard_train=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    Measurements_after=Input(rid="ri.foundry.main.dataset.51d16276-097f-4867-8e96-297bae95952e"),
    Measurements_during=Input(rid="ri.foundry.main.dataset.21288a64-d469-495e-ae9b-29d83d29a352"),
    Measurements_features_before=Input(rid="ri.foundry.main.dataset.edc9df8d-43c1-4fb4-a893-db611e8b61ea"),
    Meds_dataset_v1=Input(rid="ri.foundry.main.dataset.d2c55b64-b2ff-485c-beac-e5b5ac108940"),
    Procedure_timeindex_dataset_V1=Input(rid="ri.foundry.main.dataset.a2f26681-acd9-4fca-9a65-73a9562c7697"),
    Vitals_Dataset_V1=Input(rid="ri.foundry.main.dataset.83c4a22b-0a6a-48de-a2e9-c22ae538bd9c"),
    person_demographics=Input(rid="ri.foundry.main.dataset.f4b12b19-5966-467f-b4d8-5e4a87b3f5fd"),
    smoking_dataset_1=Input(rid="ri.foundry.main.dataset.34d2fd38-d03f-4f08-8a99-bd6617c828be"),
    utilization_updated_columns=Input(rid="ri.foundry.main.dataset.b1c6e531-a1f8-4d20-b258-b947ff462747")
)
def merged_training_data( person_demographics, Measurements_features_before, Measurements_after, Measurements_during, Vitals_Dataset_V1, Procedure_timeindex_dataset_V1, Cohort_dx_ct_features, smoking_dataset_1, utilization_updated_columns,  Long_COVID_Silver_Standard_train, Meds_dataset_v1):
   
    ## Including all tables for the modeling:
    df_pat = person_demographics.select("*").where("isTrainSet == 1")
    df_pasc = Long_COVID_Silver_Standard_train.withColumnRenamed('person_id', 'person_id_pasc')
    
    df_dxct = Cohort_dx_ct_features.withColumnRenamed('person_id', 'person_id_dxct')
    df_proc = Procedure_timeindex_dataset_V1.withColumnRenamed('person_id', 'person_id_proc')
    df_util = utilization_updated_columns.withColumnRenamed('person_id', 'person_id_utl')    
        
    df_meas_after = Measurements_after.select(*[col(c).alias("after_"+c) for c in Measurements_after.columns])
    df_meas_before = Measurements_features_before.select(*[col(c).alias("before_"+c) for c in Measurements_features_before.columns])
    df_meas_during = Measurements_during.select(*[col(c).alias("during_"+c) for c in Measurements_during.columns])
    df_vital = Vitals_Dataset_V1.withColumnRenamed('person_id', 'person_id_vital')
    df_meds = Meds_dataset_v1.withColumnRenamed('person_id', 'person_id_meds')
    df_smoke = smoking_dataset_1.withColumnRenamed('person_id', 'person_id_smoke')

    df_train = df_pat.join(df_pasc, df_pat.person_id == df_pasc.person_id_pasc, 'left') \
                        .join(df_dxct, df_pat.person_id == df_dxct.person_id_dxct, 'left') \
                        .join(df_proc, df_pat.person_id == df_proc.person_id_proc, 'left') \
                        .join(df_util, df_pat.person_id == df_util.person_id_utl, 'left') \
                        .join(df_meas_after, df_pat.person_id == df_meas_after.after_person_id, 'left') \
                        .join(df_meas_before, df_pat.person_id == df_meas_before.before_person_id, 'left') \
                        .join(df_meas_during, df_pat.person_id == df_meas_during.during_person_id, 'left') \
                        .join(df_vital, df_pat.person_id == df_vital.person_id_vital, 'left') \
                        .join(df_meds, df_pat.person_id == df_meds.person_id_meds, 'left') \
                        .join(df_smoke, df_pat.person_id == df_smoke.person_id_smoke, 'left') 

    ## Drop unnecessary columns:
    df_train = df_train.drop('person_id_pasc', 'person_id_dxct', 'person_id_proc', 'person_id_utl', 
                                'after_person_id', 'before_person_id', 'mduring_person_id', 'person_id_vital', 'person_id_meds', 'person_id_smoke', 'covid_index')

    return df_train
    
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3bc6016d-2f9c-464b-8c4e-296095919e7b"),
    merged_training_data=Input(rid="ri.foundry.main.dataset.bae2ad21-0aaa-4216-9208-c9ddc20b044d")
)
def missing_value_imputation(merged_training_data):
    
    ## Initialization: 
    df_feat = merged_training_data

    ## Addressing the missing values in different cases:
    
    # Imputation with value '0': Diagnosis Count, Procedures, Medication Count, Utilization count
    df = df_feat.na.fill(0, subset= feat_dxct + feat_proc + feat_meds + feat_utl)

    # Imputing with -1: Smoking Status,  Measurements Values  
    df = df.na.fill(-1, subset= feat_smoke + feat_vitals + feat_meas_after + feat_meas_before + feat_meas_during)
    
    return df

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4192cd64-6a5b-40c8-96d0-c95a4c570c3a"),
    merge_testing_data=Input(rid="ri.foundry.main.dataset.16159dd7-bca0-4f6d-9d90-0f8ee980241b")
)
def missing_value_imputation_test(merge_testing_data):
    
    ## Initialization: 
    df_feat = merge_testing_data

    ## Addressing the missing values in different cases:
    
    # Imputation with value '0': Diagnosis Count, Procedures, Medication Count, Utilization count
    df = df_feat.na.fill(0, subset= feat_dxct + feat_proc + feat_meds + feat_utl)

    # Imputing with -1: Smoking Status,  Measurements Values  
    df = df.na.fill(-1, subset= feat_smoke + feat_vitals + feat_meas_after + feat_meas_before + feat_meas_during)
    
    return df

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f4b12b19-5966-467f-b4d8-5e4a87b3f5fd"),
    person=Input(rid="ri.foundry.main.dataset.18101d6a-e393-4681-b03a-a4ffdadc90b2")
)
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.56726a0b-fd5f-40e4-b017-bf497bc12c0a"),
    All_procedure_with_time=Input(rid="ri.foundry.main.dataset.53510587-c350-44d1-a09d-7a3d7408a87e")
)
def procedure_time_pivoted(All_procedure_with_time):
    data = All_procedure_with_time
    pivoted_data = data.groupBy(["person_id","feature_time_index"]).pivot("concept_set_name").agg(F.count("procedure_concept_name").alias("Procedure_count")).na.fill(0)

    
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

    return pivoted_data

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b1c6e531-a1f8-4d20-b258-b947ff462747"),
    utilization_v2=Input(rid="ri.foundry.main.dataset.07b884d2-c5d8-42c1-af45-2127206c99e6")
)
def utilization_updated_columns(utilization_v2):
    df = utilization_v2
    subset_fill_0 = ['is_index_ed', 'is_index_ip', 'is_index_tele', 'is_index_op', 'before_ed_cnt', 'before_ip_cnt', 'before_op_cnt', 'during_ed_cnt', 'during_ip_cnt', 'during_op_cnt', 'after_ed_cnt', 'after_ip_cnt', 'after_op_cnt']
    subset_fill_neg_1 = ['avg_los', 'avg_icu_los']
    df = df.fillna(0, subset=subset_fill_0)
    df = df.fillna(-1, subset=subset_fill_neg_1)
    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b48e8d52-52fc-4538-b7dc-0abeed3f19e2"),
    get_training_dataset=Input(rid="ri.foundry.main.dataset.3c0351f5-f2e3-4939-bd52-c106b7c0752f")
)
def vectorizer(get_training_dataset):

    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import StandardScaler

    # Import Foundry python model wrapper
    from foundry_ml import Model, Stage

    training_data = get_training_dataset
    training_columns = features 
    
    # The passthrough option combines the numeric columns into a feature vector
    column_transformer = make_column_transformer(
        ('passthrough', training_columns)
        # (StandardScaler(), list(training_data.columns.values))
    )

    # Fit the column transformer to act as a vectorizer
    column_transformer.fit( training_data[training_columns] )

    # Wrap the vectorizer as a Stage to indicate this is the transformation that will be applied in the Model
    vectorizer = Stage(column_transformer)

    return Model(vectorizer)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6d6663d0-454f-426c-96c3-dea9f75da0e3"),
    get_training_dataset=Input(rid="ri.foundry.main.dataset.3c0351f5-f2e3-4939-bd52-c106b7c0752f"),
    vectorizer=Input(rid="ri.foundry.main.dataset.b48e8d52-52fc-4538-b7dc-0abeed3f19e2")
)
def xgb_train_test_model(vectorizer, get_training_dataset):

    ## Library Import: 
    import xgboost as xgb

    ## Initializations:
    # Import the training dataset in Pandas Format
    # Vectorizer in the object input type:
    training_data = get_training_dataset

    # Applies vectorizer to produce a DataFrame with all original columns and the column of vectorized data: 
    training_df = vectorizer.transform(training_data[features])

    # Invoke a palantir helper function to convert a column of vectors into a NumPy matrix and handle sparsity
    X = extract_matrix(training_df, 'features')
    y = training_data[label]

     # Train a XGBoost model 
    clf =  xgb.XGBClassifier(max_depth = 8, min_child_weight = 1, colsample_bytree= 0.5, learning_rate= 0.1,   reg_alpha = 0.25, reg_lambda= 1.2, scale_pos_weight= 5, random_state= random_seed) 
    clf.fit(X, y)

    # Return Model object that now contains a pipeline of transformations
    model = Model(vectorizer, Stage(clf, input_column_name='features'))

    return model

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ef243559-648d-4247-a286-7bcc2af70e9a"),
    get_testing_dataset=Input(rid="ri.foundry.main.dataset.3cd10bcd-fead-4854-8029-b1005944e679"),
    xgb_train_test_model=Input(rid="ri.foundry.main.dataset.6d6663d0-454f-426c-96c3-dea9f75da0e3")
)
'''
## IMPORTANT: 
1. Importing the data as a transform output, which also carries metadata with it, ensuring that the model was evaluated on the correct version of the dataset.
2. training_data.dataframe().toPandas() syntax is to read transform input as a DataFrame
'''

def xgb_validation_evaluation( xgb_train_test_model, get_testing_dataset):
    return calculate_evaluation_metrics(model=xgb_train_test_model, data=get_testing_dataset, threshold=0.5)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d4fa315e-de39-42ca-885c-ed02bf71e7bc"),
    get_training_dataset=Input(rid="ri.foundry.main.dataset.3c0351f5-f2e3-4939-bd52-c106b7c0752f"),
    xgb_train_test_model=Input(rid="ri.foundry.main.dataset.6d6663d0-454f-426c-96c3-dea9f75da0e3")
)
'''
## IMPORTANT: 
1. Importing the data as a transform output, which also carries metadata with it, ensuring that the model was evaluated on the correct version of the dataset.
2. training_data.dataframe().toPandas() syntax is to read transform input as a DataFrame
'''

def xgboost_training_evaluation(xgb_train_test_model, get_training_dataset):
    return calculate_evaluation_metrics(model=xgb_train_test_model, data=get_training_dataset, threshold=0.5)

