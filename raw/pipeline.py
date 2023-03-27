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

