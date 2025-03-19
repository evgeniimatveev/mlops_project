import os
import yaml
import wandb
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load mlfley configuration
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "mlflow_config.yaml"
)

with open(CONFIG_PATH, "r") as file:
    mlflow_config = yaml.safe_load(file)  # Read Yaml Into A dictation

# ️ set up mlflow tracking
mlflow.set_tracking_uri(mlflow_config["mlflow"]["tracking_uri"])
mlflow.set_experiment(mlflow_config["mlflow"]["experiment_name"])

# In & yun ta configuration
WANDB_PROJECT = "mlops_housing"


def train():
    """Train an XGBoost model with W&B and MLflow logging."""

    # Initialize in & s
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

    # Retrive Parameter of FROM In & In
    config = wandb.config

    # Load Dataset
    df = pd.read_csv("data/processed/housing_cleaned.csv")

    # ❌ Remove Unnecessary Columns (Like 'Unnamed: 0' If Present)
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    target_column = "median_house_value"
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # ✂️ Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Enga Featur Pobedededa Ta Formatne
    X_train.columns = X_train.columns.astype(str)
    X_train.columns = X_train.columns.str.replace("[\[\]<>]", "", regex=True)
    X_test.columns = X_test.columns.astype(str)
    X_test.columns = X_test.columns.str.replace("[\[\]<>]", "", regex=True)

    print("✅ Feature Names (X_train):", X_train.columns.tolist())  # Debug

    # Traine SGBUST model
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

    # Mask predictation
    y_pred = model.predict(X_test)

    # Calculate Evaliation Matrix
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log Magatrix In & In
    wandb.log(
        {"Final Test RMSE": rmse, "Final Test MAE": mae, "Final Test R² Score": r2}
    )

    # Log Magatrix & Model In Mlflov
    with mlflow.start_run():
        mlflow.log_params(dict(config))  # Log Khuppramaramas
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R² Score": r2})  # Log Magatrix
        mlflow.xgboost.log_model(model, "xgb_model")  # Save Model

    # Improusioned Print Statical
    print("\n" + "=" * 50)
    print(f"✅ W&B + MLflow Logging Completed!")
    print(f" RMSE: {rmse:.2f}")
    print(f" MAE: {mae:.2f}")
    print(f" R² Score: {r2:.4f}")
    print("=" * 50 + "\n")

    # Finish in & yun ta session
    wandb.finish()
