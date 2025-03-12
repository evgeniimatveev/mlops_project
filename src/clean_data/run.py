import os  # For file and directory handling
import pandas as pd  # For working with CSV files
import wandb  # Weights & Biases for tracking and logging

# Initialize Weights & Biases for experiment tracking
wandb.init(project="mlops_housing", name="clean_data")

# Get absolute path to the project's root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Define correct paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "housing.csv")  # Path to raw data
PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)  # Directory for cleaned data
CLEAN_DATA_PATH = os.path.join(
    PROCESSED_DIR, "housing_cleaned.csv"
)  # Path for cleaned dataset

# Ensure the processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Log start of data cleaning
wandb.log({"status": "Loading dataset..."})
print(f"📂 Loading dataset from: {RAW_DATA_PATH}")

# Load dataset
df = pd.read_csv(RAW_DATA_PATH)

# Log initial shape of the dataset
wandb.log({"initial_rows": df.shape[0], "initial_columns": df.shape[1]})
print(f"🔍 Initial dataset shape: {df.shape}")

# Remove duplicate rows
df.drop_duplicates(inplace=True)
wandb.log({"rows_after_deduplication": df.shape[0]})
print(f"✅ Removed duplicates. New shape: {df.shape}")

# Handle missing values: fill numeric columns with their mean
df.fillna(df.mean(numeric_only=True), inplace=True)
wandb.log({"missing_values_after_cleaning": df.isnull().sum().sum()})
print(f"✅ Filled missing values.")

# Convert categorical "ocean_proximity" into numerical categories
if "ocean_proximity" in df.columns:
    df = pd.get_dummies(df, columns=["ocean_proximity"])
    print(f"✅ Encoded categorical column 'ocean_proximity'.")

# Save cleaned dataset
df.to_csv(CLEAN_DATA_PATH, index=False)
print(f"💾 Cleaned dataset saved to: {CLEAN_DATA_PATH}")

# Log final dataset shape and completion message
wandb.log(
    {
        "final_rows": df.shape[0],
        "final_columns": df.shape[1],
        "status": "Data cleaning completed successfully!",
    }
)
print("✅ Data cleaning completed successfully!")
