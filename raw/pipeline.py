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

