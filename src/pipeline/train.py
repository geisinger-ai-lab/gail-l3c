import argparse
import os
from typing import Text
from webbrowser import get

import pandas as pd
import xgboost as xgb
import yaml
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

from src.common import features, get_logger, label


def get_vectorizer(get_training_dataset):

    training_data = get_training_dataset
    training_columns = features

    # The passthrough option combines the numeric columns into a feature vector
    column_transformer = make_column_transformer(
        ("passthrough", training_columns)
        # (StandardScaler(), list(training_data.columns.values))
    )

    # Fit the column transformer to act as a vectorizer
    column_transformer.fit(training_data[training_columns])

    return column_transformer


def train_xgb_model(vectorizer, get_training_dataset, config):

    ## Initializations:
    # Import the training dataset in Pandas Format
    # Vectorizer in the object input type:
    training_data = get_training_dataset

    # Applies vectorizer to produce a DataFrame with all original columns and the column of vectorized data:
    training_df = vectorizer.transform(training_data[features])

    X = training_df
    y = training_data[label]

    # Train a XGBoost model
    # TODO move these to the config?
    random_seed = config["train"].get("random_seed")
    clf = xgb.XGBClassifier(
        max_depth=8,
        min_child_weight=1,
        colsample_bytree=0.5,
        learning_rate=0.1,
        reg_alpha=0.25,
        reg_lambda=1.2,
        scale_pos_weight=5,
        random_state=random_seed,
    )
    clf.fit(X, y)

    return clf


def train(config_path: Text) -> None:
    """Train
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("TRAIN", log_level=config["common"]["log_level"])

    logger.info("Training...")

    # Load data
    featurized_df_path = os.path.join(
        config["featurize"]["data_path_featurized"], "featurized_data_train.csv"
    )
    featurized_df = pd.read_csv(featurized_df_path)

    # Initialize the vectorizer
    vectorizer = get_vectorizer(featurized_df)

    # Train the model
    clf = train_xgb_model(vectorizer, featurized_df, config)

    # Save the model locally
    model_path = os.path.join(config["train"]["model_path"], config["train"]["model_name"])
    clf.save_model(model_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    clf = train(config_path=args.config)
