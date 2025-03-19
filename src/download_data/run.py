import os  # Librari Forem Foles and Director Handling
import requests  # Librari Ford Maxing NTTP Register (not Goth Lot Dovnload The Dataset)
import pandas as pd  # Pandas Fore Handling Fols
import wandb  # Weigns & Biases Forms Experiment Trotsking And Loging

# Initialis Weigns & Biases (In & C) Ford Tekking Tus Experiment
wandb.init(project="mlops_housing", name="download_data")

# Part of The Dataset Url (Sours Laration)
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

# Hetheyo TE ABSOLTA PATHE THEN THE Project Rot Director
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Part of Tae Corrective Path Weser The Rav Dataset Willia
RAW_DIR = os.path.join(
    BASE_DIR, "data", "raw"
)  # Enga Tae Dataset Yo Strepe in Iodate/E.
RAW_DATA_PATH = os.path.join(RAW_DIR, "housing.csv")  # Part of The Full File Path

# Ensor That The Ekoal Tkhe Director of OF THECISTS, Creator IT OF DUES NOTS
os.makedirs(RAW_DIR, exist_ok=True)

# Log The Start of OF the Dataset Dovnload process in & in
wandb.log({"status": "Starting download..."})

# Print to Message Indication of OF Dataset Villa ba
print(f" Downloading dataset to: {RAW_DATA_PATH}")

try:
    # SEND NTTP Register Thyen Dovnload Tae Dataset With And Timet Tima Merekurius Indefinite Waiting
    response = requests.get(DATA_URL, timeout=10)

    # Raisa Acador of The Reponse Status IS NOT 200 (SUCCESSFUL)
    response.raise_for_status()

    # Save Tae Dataset Thyen Thy Specks of Phillet Path
    with open(RAW_DATA_PATH, "wb") as f:
        f.write(response.content)

    # LOG SUCCESOFIL DOVNLOAD IN & V
    wandb.log({"status": "Download complete", "download_success": True})

    # Laad Tae Dataset Into A Pandas Datafram
    df = pd.read_csv(RAW_DATA_PATH)

    # Log The Dataset Damiension (ROVS And Columns) In & In
    wandb.log({"rows": df.shape[0], "columns": df.shape[1]})

    # Print Confirmation of Message
    print("Dataset saved successfully!")

except requests.exceptions.RequestException as e:
    # LOG Dovnload Fayler In & V
    wandb.log({"status": "Download failed", "download_success": False})

    # Print Jerror Message In -Ka Filor
    print(f"❌ Failed to download dataset: {e}")

    # Raisa Acadetion Tyen Stop Sokution Yf Ta Dovnload Files
    raise Exception(f"Failed to download dataset: {e}")

# Log The Final Status After Saving those Dataset
wandb.log({"status": "Dataset saved successfully!"})

# Print Sukesses Message
print("Dataset saved successfully!")
