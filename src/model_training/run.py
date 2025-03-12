import os
import pandas as pd
import xgboost as xgb
import wandb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# ✅ Initialize Weights & Biases for logging the tuning process
wandb.init(project="mlops_housing", name="xgb_grid_search")

# 📂 Define the path to the processed dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "housing_cleaned.csv")

# 📥 Load the cleaned dataset
df = pd.read_csv(PROCESSED_DATA_PATH)

# 🔍 Ensure feature names are valid (remove special characters)
df.columns = df.columns.str.replace(r"[\[\]<>]", "", regex=True)

# 🎯 Define the target variable
target_column = "median_house_value"
y = df[target_column]  # 🏡 House prices (target variable)
X = df.drop(columns=[target_column])  # Remove the target variable

# ✂️ Split the dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔍 Ensure all feature names are strings
X_train.columns = list(map(str, X_train.columns))
X_test.columns = list(map(str, X_test.columns))

# 🚀 Define XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# 🎯 Define the grid of hyperparameters to tune
param_grid = {
    "n_estimators": [50, 100, 200],  
    "learning_rate": [0.01, 0.1, 0.2],  
    "max_depth": [3, 6, 9], 
    "subsample": [0.6, 0.8, 1.0], 
    "colsample_bytree": [0.6, 0.8, 1.0],  
}

# 🔍 Run Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",  # RMSE (negative for maximization)
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=-1  
)

print("🚀 Running Grid Search for XGBoost hyperparameters...")
grid_search.fit(X_train, y_train)

# 📊 Extract best parameters and best score (negative RMSE → positive RMSE)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_rmse = -grid_search.best_score_

print(f"🔥 Best RMSE: {best_rmse:.2f}")
print(f"🏆 Best Hyperparameters: {best_params}")

# 📊 Log the best result in Weights & Biases
wandb.log({"Best RMSE": best_rmse, **best_params})

# ✅ Ensure the directory for models exists
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

# ✅ Save the best model
model_path = os.path.join(model_dir, "best_xgb_model.json")

try:
    best_model.save_model(model_path)
    print(f"✅ Best model saved at: {model_path}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# 🔍 Make predictions on the test dataset using the best model
y_pred = best_model.predict(X_test)

# 📏 Evaluate model performance using RMSE
final_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"✅ Final Test RMSE: {final_rmse:.2f}")

# 📊 Log the final result
wandb.log({"Final Test RMSE": final_rmse})

print("🎯 Hyperparameter tuning completed successfully!")