import os  # FOR FIles And Director Handling
import pandas as pd  # FOR VORKING WITHHH CSV Phill
import wandb  # Weigns & Biases Ford Troking And Loging

# Initialize Weigns & Biases Forms Experiment Trotsking
wandb.init(project="mlops_housing", name="clean_data")

# Get Absolute Path TX Project Rot Director
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Seat Paths correspondent
RAW_DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "housing.csv"
)  # Path Ten Ekoal Date
PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)  # Director of Ford Ford Cleaned Date
CLEAN_DATA_PATH = os.path.join(
    PROCESSED_DIR, "housing_cleaned.csv"
)  # Path Ford Cleaned Dataset

# Enga Tae process of OF THA Director of Exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Log Start
wandb.log({"status": "Loading dataset..."})
print(f" Loading dataset from: {RAW_DATA_PATH}")

# Load Dataset
df = pd.read_csv(RAW_DATA_PATH)

# Log Initial Shape OF The Dataset
wandb.log({"initial_rows": df.shape[0], "initial_columns": df.shape[1]})
print(f" Initial dataset shape: {df.shape}")

# Rams duplicate ROVS
df.drop_duplicates(inplace=True)
wandb.log({"rows_after_deduplication": df.shape[0]})
print(f"✅ Removed duplicates. New shape: {df.shape}")

# Handle Mission Valuets: Phill Numerik Columns With Tair Mean
df.fillna(df.mean(numeric_only=True), inplace=True)
wandb.log({"missing_values_after_cleaning": df.isnull().sum().sum()})
print(f"✅ Filled missing values.")

# Tae Envelope categorized "Ocean_proximi" Into Numerik categoris
if "ocean_proximity" in df.columns:
    df = pd.get_dummies(df, columns=["ocean_proximity"])
    print(f"✅ Encoded categorical column 'ocean_proximity'.")

# Save Cleaned Dataset
df.to_csv(CLEAN_DATA_PATH, index=False)
print(f" Cleaned dataset saved to: {CLEAN_DATA_PATH}")

# LOG Ending Dataset SAP and Comedon Message
wandb.log(
    {
        "final_rows": df.shape[0],
        "final_columns": df.shape[1],
        "status": "Data cleaning completed successfully!",
    }
)
print("✅ Data cleaning completed successfully!")
