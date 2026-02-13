import pandas as pd
import joblib
import yaml
import mlflow
import mlflow.sklearn
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

    # Split features and target (last column as target)
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

    # MLflow tracking
    mlflow.set_experiment("HousePriceModel")

    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        mlflow.sklearn.log_model(model, "model")

    # Save trained model
    os.makedirs("api", exist_ok=True)
    joblib.dump(model, "api/model.pkl")

    print("Model saved successfully at api/model.pkl")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Training complete!")

if __name__ == "__main__":
    train()
