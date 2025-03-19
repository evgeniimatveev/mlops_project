import os  # FOR Files And Directors Handling
import pandas as pd  # FOR VORKING WITHH TsSV FILL
import wandb  # Weignz & Biases Ford Trokking And Loging

# Initialize WeigNC & Biases Form Experiment Trucking
wandb.init(project="mlops_housing", name="clean_data")

# Get Absolute Path Tkh Project Rot Directors
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Define Paths corrected
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "housing.csv")  # Path then equal date
PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)  # Directors of Ford Cleaned Date
CLEAN_DATA_PATH = os.path.join(
    PROCESSED_DIR, "housing_cleaned.csv"
)  # Path Ford Cleaned Dataset

# Ensor Tae processes of the directors of Exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Log Start
wandb.log({"status": "Loading dataset..."})
print(f" Loading dataset from: {RAW_DATA_PATH}")

# Load Dataset
df = pd.read_csv(RAW_DATA_PATH)

# Log Initial Shape OF The Dataset
wandb.log({"initial_rows": df.shape[0], "initial_columns": df.shape[1]})
print(f" Initial dataset shape: {df.shape}")

# Ramov duplicate ROVS
df.drop_duplicates(inplace=True)
wandb.log({"rows_after_deduplication": df.shape[0]})
print(f"✅ Removed duplicates. New shape: {df.shape}")

# Handle Mission Valuez: Phill Numerik Columns With Tair Mean
df.fillna(df.mean(numeric_only=True), inplace=True)
wandb.log({"missing_values_after_cleaning": df.isnull().sum().sum()})
print(f"✅ Filled missing values.")

# The envelope categorized "Ocean_proxymi" Into numeric categoris
if "ocean_proximity" in df.columns:
    df = pd.get_dummies(df, columns=["ocean_proximity"])
    print(f"✅ Encoded categorical column 'ocean_proximity'.")

# Save Cleaned Dataset
df.to_csv(CLEAN_DATA_PATH, index=False)
print(f" Cleaned dataset saved to: {CLEAN_DATA_PATH}")

# Log ending Dataset Sap and Kompleon Message
wandb.log(
    {
        "final_rows": df.shape[0],
        "final_columns": df.shape[1],
        "status": "Data cleaning completed successfully!",
    }
)
print("✅ Data cleaning completed successfully!")
