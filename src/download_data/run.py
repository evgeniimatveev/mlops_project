`import os  # Library for file and directory handling
import requests  # Library for making HTTP requests (used to download the dataset)
import pandas as pd  # Pandas for handling CSV files
import wandb  # Weights & Biases for experiment tracking and logging

# Initialize Weights & Biases (W&B) for tracking the experiment
wandb.init(project="mlops_housing", name="download_data")

# Define the dataset URL (source location)
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

# Get the absolute path to the project's root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Define the correct path where the raw dataset will be saved
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")  # Ensure the dataset is stored in `data/raw/`
RAW_DATA_PATH = os.path.join(RAW_DIR, "housing.csv")  # Define the full file path

# Ensure that the raw data directory exists, create it if it does not
os.makedirs(RAW_DIR, exist_ok=True)

# Log the start of the dataset download process in W&B
wandb.log({"status": "Starting download..."})

# Print message indicating where the dataset will be saved
print(f"📥 Downloading dataset to: {RAW_DATA_PATH}")

try:
    # Send an HTTP request to download the dataset with a timeout to prevent indefinite waiting
    response = requests.get(DATA_URL, timeout=10)
    
    # Raise an error if the response status is not 200 (successful)
    response.raise_for_status()

    # Save the dataset to the specified file path
    with open(RAW_DATA_PATH, "wb") as f:
        f.write(response.content)

    # Log successful download in W&B
    wandb.log({"status": "Download complete", "download_success": True})

    # Load the dataset into a Pandas DataFrame
    df = pd.read_csv(RAW_DATA_PATH)

    # Log the dataset dimensions (rows and columns) in W&B
    wandb.log({"rows": df.shape[0], "columns": df.shape[1]})

    # Print confirmation message
    print("✅ Dataset downloaded successfully!")

except requests.exceptions.RequestException as e:
    # Log download failure in W&B
    wandb.log({"status": "Download failed", "download_success": False})

    # Print error message in case of a failure
    print(f"❌ Failed to download dataset: {e}")

    # Raise an exception to stop execution if the download fails
    raise Exception(f"Failed to download dataset: {e}")

# Log the final status after saving the dataset
wandb.log({"status": "Dataset saved successfully!"})

# Print success message
print("✅ Dataset saved successfully!")`