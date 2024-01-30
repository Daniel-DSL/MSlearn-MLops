# Voor de nieuwe branch!

# Import libraries

import argparse
import glob
import os

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# define functions
def main(args):
    print(f"Training data path: {args.training_data}")

    # TO DO: enable autologging
    mlflow.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")

    # Read the single CSV file
    if os.path.isfile(path):
        return pd.read_csv(path)

    raise RuntimeError(f"No CSV files found in provided data path: {path}")


# To Do Done: add function to split data
def split_data(df):
    """Function to split the data into X and Y variables"""
    X, y = (
        df[
            [
                "Pregnancies",
                "PlasmaGlucose",
                "DiastolicBloodPressure",
                "TricepsThickness",
                "SerumInsulin",
                "BMI",
                "DiabetesPedigree",
                "Age",
            ]
        ].values,
        df["Diabetic"].values,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # start MLflow run
    with mlflow.start_run():
        # train model
        model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
        model.fit(X_train, y_train)

        # log model parameters
        mlflow.log_params({"C": 1 / reg_rate, "solver": "liblinear"})

        # log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # log the trained model
        mlflow.sklearn.log_model(model, "logistic_regression_model")


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--training_data", dest="training_data", type=str, required=True
    )
    parser.add_argument("--reg_rate", dest="reg_rate", type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
