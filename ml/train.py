import pandas as pd
import joblib
import yaml
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os


def train():
    print("Training model started...")

    # Load params
    with open("ml/params.yml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    # Load dataset
    df = pd.read_csv("data/data.csv")

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # MLflow tracking (safe for Linux runner)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("HousePriceModel")

    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

    # Save trained model for FastAPI
    joblib.dump(model, "api/model.pkl")

    print("Model saved at api/model.pkl")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Training complete!")


if __name__ == "__main__":
    train()
