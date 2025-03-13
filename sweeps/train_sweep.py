import os
import yaml
import wandb
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#  Load MLflow configuration
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "mlflow_config.yaml"
)

with open(CONFIG_PATH, "r") as file:
    mlflow_config = yaml.safe_load(file)  # Read YAML into a dictionary

# ️ Set up MLflow tracking
mlflow.set_tracking_uri(mlflow_config["mlflow"]["tracking_uri"])
mlflow.set_experiment(mlflow_config["mlflow"]["experiment_name"])

#  W&B configuration
WANDB_PROJECT = "mlops_housing"


def train():
    """Train an XGBoost model with W&B and MLflow logging."""

    #  Initialize W&B
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    )

    #  Retrieve parameters from W&B
    config = wandb.config

    #  Load dataset
    df = pd.read_csv("data/processed/housing_cleaned.csv")

    # ❌ Remove unnecessary columns (like 'Unnamed: 0' if present)
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    target_column = "median_house_value"
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # ✂️ Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #  Ensure feature names are properly formatted
    X_train.columns = X_train.columns.astype(str)
    X_train.columns = X_train.columns.str.replace("[\[\]<>]", "", regex=True)
    X_test.columns = X_test.columns.astype(str)
    X_test.columns = X_test.columns.str.replace("[\[\]<>]", "", regex=True)

    print("✅ Feature Names (X_train):", X_train.columns.tolist())  # Debug

    #  Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    #  Make predictions
    y_pred = model.predict(X_test)

    #  Calculate evaluation metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #  Log metrics in W&B
    wandb.log(
        {"Final Test RMSE": rmse, "Final Test MAE": mae, "Final Test R² Score": r2}
    )

    #  Log metrics & model in MLflow
    with mlflow.start_run():
        mlflow.log_params(dict(config))  # Log hyperparameters
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R² Score": r2})  # Log metrics
        mlflow.xgboost.log_model(model, "xgb_model")  # Save model

    #  Improved print statement
    print("\n" + "=" * 50)
    print(f"✅ W&B + MLflow Logging Completed!")
    print(f" RMSE: {rmse:.2f}")
    print(f" MAE: {mae:.2f}")
    print(f" R² Score: {r2:.4f}")
    print("=" * 50 + "\n")

    #  Finish W&B session
    wandb.finish()
