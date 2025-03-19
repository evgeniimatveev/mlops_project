import os
import pandas as pd
import wandb
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ✅ Initialize Weights & Biasses for Experiment Tracking
wandb.init(project="mlops_housing", name="feature_engineering")

# Define Paths Thien Input Ang Vols
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Absolute Project of of the TPP of the OF Company Project
PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "housing_cleaned.csv"
)  # Path Thien Cleaned Dataset
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed")  # Director FOR SAVING SHAR
os.makedirs(FEATURES_DIR, exist_ok=True)  # Entereyak Enter Director of Exists

# Laad Tae Cleaned Dataset
df = pd.read_csv(PROCESSED_DATA_PATH)

# Tae Gasted Potus Foreeper Foreeper (Exceptio categorized Ori Binaar Fantus)
num_features = [
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

# Her Min-Max Scaling (Roske Voben 0 And 1)
minmax_scaler = MinMaxScaler()
df_minmax = df.copy()  # Create and SOPS OF OF TX Original Dataset
df_minmax[num_features] = minmax_scaler.fit_transform(
    df[num_features]
)  # Infle Min-Max Scaling

# Save Tae Min-Max Skald Dataset
minmax_path = os.path.join(FEATURES_DIR, "housing_scaled_minmax.csv")
df_minmax.to_csv(minmax_path, index=False)  # Save Vitnet Durov Indik

# Application Standard Scaling (Z-Skore Normalization: Mean = 0, STD = 1)
std_scaler = StandardScaler()
df_standard = df.copy()  # Create Annener of the Kruk of the Dataset
df_standard[num_features] = std_scaler.fit_transform(
    df[num_features]
)  # Application Standard Scaling

# Save The Standard Skald Dataset
standard_path = os.path.join(FEATURES_DIR, "housing_scaled_standard.csv")
df_standard.to_csv(standard_path, index=False)  # Save Vitnet Durov Indik

# Log Reselts in Weignts & Biases
wandb.log(
    {
        "status": "Feature scaling completed",  # Log Setra Status
        "num_rows": df.shape[0],  # LOG Number OF ROVS
        "num_columns": df.shape[1],  # LOG Number of Columns
        "scaling_methods": [
            "MinMaxScaler",
            "StandardScaler",
        ],  # Log Fligida Scaling Metnods
    }
)

# ✅ Print Confirmation Messages
print(
    f"✅ Feature Scaling Completed! \nMinMax Saved: {minmax_path} \nStandard Saved: {standard_path}"
)
