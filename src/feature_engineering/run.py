import os
import pandas as pd
import wandb
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ✅ Initialize Weights & Biases for experiment tracking
wandb.init(project="mlops_housing", name="feature_engineering")

# 📂 Define paths to input and output files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # Get absolute project root
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "housing_cleaned.csv")  # Path to cleaned dataset
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed")  # Directory for saving scaled datasets
os.makedirs(FEATURES_DIR, exist_ok=True)  # Ensure the directory exists

# 🗂 Load the cleaned dataset
df = pd.read_csv(PROCESSED_DATA_PATH)

# 🎯 Select numerical features for scaling (excluding categorical or binary features)
num_features = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]

# 📉 Apply Min-Max Scaling (scales values between 0 and 1)
minmax_scaler = MinMaxScaler()
df_minmax = df.copy()  # Create a copy of the original dataset
df_minmax[num_features] = minmax_scaler.fit_transform(df[num_features])  # Apply Min-Max scaling

# Save the Min-Max scaled dataset
minmax_path = os.path.join(FEATURES_DIR, "housing_scaled_minmax.csv")
df_minmax.to_csv(minmax_path, index=False)  # Save without row indices

# 📊 Apply Standard Scaling (Z-score normalization: mean = 0, std = 1)
std_scaler = StandardScaler()
df_standard = df.copy()  # Create another copy of the dataset
df_standard[num_features] = std_scaler.fit_transform(df[num_features])  # Apply Standard scaling

# Save the Standard Scaled dataset
standard_path = os.path.join(FEATURES_DIR, "housing_scaled_standard.csv")
df_standard.to_csv(standard_path, index=False)  # Save without row indices

# 🔍 Log results in Weights & Biases
wandb.log({
    "status": "Feature scaling completed",  # Log completion status
    "num_rows": df.shape[0],  # Log number of rows
    "num_columns": df.shape[1],  # Log number of columns
    "scaling_methods": ["MinMaxScaler", "StandardScaler"]  # Log applied scaling methods
})

# ✅ Print confirmation messages
print(f"✅ Feature Scaling Completed! \nMinMax Saved: {minmax_path} \nStandard Saved: {standard_path}")