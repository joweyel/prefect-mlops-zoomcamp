import os
import requests
import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.models import infer_signature
import xgboost as xgb
from prefect import flow, task
from typing import Tuple


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


@task
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[
    scipy.sparse._csr.csr_matrix,
    scipy.sparse._csr.csr_matrix,
    np.ndarray,
    np.ndarray,
    sklearn.feature_extraction.DictVectorizer,
]:
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:squarederror",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        y_pred = booster.predict(valid)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Infer signature using raw input features (before DMatrix conversion)
        signature = infer_signature(X_val.toarray(), y_val)
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow", signature=signature)
    return None

@task(name="download_data", retries=3, retry_delay_seconds=5)
def download_data(
    train_path: str = "./data/green_tripdata_2021-01.parquet",
    val_path: str = "./data/green_tripdata_2021-02.parquet",
) -> bool:
    """Download dataset if missing. Returns True if data is obtained (downloaded or already available)."""
    
    base_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    train_url: str = base_url + train_path.split("/")[-1]
    val_url: str = base_url + val_path.split("/")[-1]

    data_obtained = True  # Start with True, will turn False if any download fails

    for path, url in [(train_path, train_url), (val_path, val_url)]:
        if not os.path.exists(path):  # Download only if missing
            print(f"Downloading: {url}")
            response = requests.get(url, stream=True)

            if response.status_code == 200:  # Correct status check (integer, not string)
                with open(path, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {url}, HTTP {response.status_code}")
                data_obtained = False  # Mark failure if a download fails
        else:
            print(f"File already exists: {path}")

    return data_obtained


@flow(name="main-flow")
def main_flow(
    train_path: str = "./data/green_tripdata_2021-01.parquet",
    val_path: str = "./data/green_tripdata_2021-02.parquet",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    # Load
    download_data(train_path, val_path)
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv)


if __name__ == "__main__":
    # Updated file names
    train_path: str = "./data/green_tripdata_2023-01.parquet"
    val_path: str = "./data/green_tripdata_2023-02.parquet"
    main_flow(train_path, val_path)
