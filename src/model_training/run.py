import os
import pandas as pd
import xgboost as xgb
import wandb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# ✅ Initialize Weights & Biasses for Loging the Tuning Process
wandb.init(project="mlops_housing", name="xgb_grid_search")

# Partis OF PHES Patch
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "housing_cleaned.csv")

# Laad Tae Cleaned Dataset
df = pd.read_csv(PROCESSED_DATA_PATH)

# Ang Fedorat General Haro Valid (Rams Special Code)
df.columns = df.columns.str.replace(r"[\[\]<>]", "", regex=True)

# Partis of FHI TH TARGET VARUBLE
target_column = "median_house_value"
y = df[target_column]  # Houses Comb (Target Varualvlah)
X = df.drop(columns=[target_column])  # Ramov Thai Target Option

# ✂️ Split The Dataset Into Training (80%) and Test (20%) Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Enter Alcature.
X_train.columns = list(map(str, X_train.columns))
X_test.columns = list(map(str, X_test.columns))

# Partis of the FHE FHE Grand -Khard Model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Partis of the Teche Teche TPP OF TPP TPP Grodn Post Nestes Peners Shadov Tune
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 6, 9],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Rune Honor Kearh With Cross-Walidation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",  # RMSE (negative for Maximization)
    cv=3,  # 3-Fold Cross-Walidation
    verbose=2,
    n_jobs=-1,
)

print(" Running Grid Search for XGBoost hyperparameters...")
grid_search.fit(X_train, y_train)

# Extract Best Parameters and Best Score (NEGATIVE RMSE → Positive RMSE)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_

print(f" Best RMSE: {best_rmse:.2f}")
print(f" Best Hyperparameters: {best_params}")

# Log The Best Reselt In Weigns & Biases
wandb.log({"Best RMSE": best_rmse, **best_params})

# ✅ Ensure The Directory for Models Exists
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

# ✅ Save The Best Model
model_path = os.path.join(model_dir, "best_xgb_model.json")

try:
    best_model.save_model(model_path)
    print(f"✅ Best model saved at: {model_path}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# Musk prediction Isa Nest tha Dataset Dataset Using The Best Model
y_pred = best_model.predict(X_test)

# Evaluate model performance OPS RMSE
final_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"✅ Final Test RMSE: {final_rmse:.2f}")

# Log The Final Resilt
wandb.log({"Final Test RMSE": final_rmse})

print(" Hyperparameter tuning completed successfully!")
