import argparse
import os
from typing import Text

import pandas as pd
import xgboost as xgb
import yaml

from src.common import calculate_evaluation_metrics, features, get_logger


def infer(config_path: Text) -> None:
    """Infer - Loads the model and featurized data file(s);
    makes predictions and saves predicted probabilities to csv;
    prints a summary of performance metrics for each file to the log

    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("INFER", log_level=config["common"]["log_level"])

    logger.info("Generating inferences...")

    # Load the model
    model_path = os.path.join(config["train"]["model_path"], config["train"]["model_name"])
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)

    threshold = config["infer"]["threshold"]

    # Load data for train, test, etc.
    for data_file_name in config["infer"]["data_files"]:
        logger.info(f"Processing {data_file_name}...")

        featurized_df_path = os.path.join(
            config["featurize"]["data_path_featurized"], data_file_name
        )
        featurized_df = pd.read_csv(featurized_df_path)

        # Predict
        prediction_df = featurized_df[["person_id"]].copy()
        prediction_df["predict_prob"] = xgb_model.predict_proba(featurized_df[features])[:, 1]

        # Save predictions (person_id, prob, pred_label)
        predictions_path = config["infer"]["predictions_path"]
        if not os.path.exists(predictions_path):
            os.mkdir(predictions_path)
        predictions_csv = os.path.join(predictions_path, data_file_name + "-predictions.csv")
        logger.info(f"Saving predicted probabilities to {predictions_path}...")
        prediction_df.to_csv(predictions_csv, index=False)

        # Print evaluation metrics
        # TODO: make a nicer printed summary and save plots to png
        logger.info(f"Performance metrics summary:")
        metrics_dict = calculate_evaluation_metrics(xgb_model, featurized_df, threshold=threshold)
        logger.info(f"AUC: {metrics_dict['area_under_roc']}")
        logger.info(f"average_precision: {metrics_dict['average_precision']}")
        logger.info(f"brier_score: {metrics_dict['brier_score']}")
        logger.info(f"brier_score: {metrics_dict['brier_score']}")
        logger.info(f"f1_score@{threshold}: {metrics_dict[f'f1_score@{threshold}']}")
        logger.info(f"precision@{threshold}: {metrics_dict[f'precision@{threshold}']}")
        logger.info(f"recall@{threshold}: {metrics_dict[f'recall@{threshold}']}")
        logger.info("---" * 12)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    infer(config_path=args.config)
