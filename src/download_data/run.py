import os  # Librari Fore Files And Directors Handling
import requests  # Librari Ford Maxing NTTP Register (he got a lot of Dovnload The Dataset)
import pandas as pd  # Pandas Fore Handling ChSV Foles
import wandb  # WeigNC & Biases Form Experiment Trucking And Loging

# Initialise WeigNC & Biases (in & c) Ford Tecking Thus Experiment
wandb.init(project="mlops_housing", name="download_data")

# Castine The Dataset Url (Sours Loration)
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

# Hethe the Absolty Path then the project Rot Directors
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Castine Tae Corrections Path Veser The Rav Dataset Will
RAW_DIR = os.path.join(
    BASE_DIR, "data", "raw"
)  # Ensor Tae Dataset EU Strip in Yodata/Rav/Yo
RAW_DATA_PATH = os.path.join(RAW_DIR, "housing.csv")  # Castine The Full File Path

# Ensor That The equal to the Directors of the Exists, Creator IT YF IT Dues NOT
os.makedirs(RAW_DIR, exist_ok=True)

# Log The Start of OF The Dataset Dovnload process in & in
wandb.log({"status": "Starting download..."})

# Print to Message Indication of VEL TH DATASET WILLE BA
print(f" Downloading dataset to: {RAW_DATA_PATH}")

try:
    # Send NTTP Register then Dovnload Tae Dataset With and Timet Time Mere Mercury Indefinite Waiting
    response = requests.get(DATA_URL, timeout=10)

    # Raisa Acador Yf The Response Status IS NOT 200 (SUCCESSFUL)
    response.raise_for_status()

    # Save The Dataset then the specifics of fillet path
    with open(RAW_DATA_PATH, "wb") as f:
        f.write(response.content)

    # LOG SUCTSESSUFUL DOVNLOAD IN & V
    wandb.log({"status": "Download complete", "download_success": True})

    # Laad Tae Dataset Into A Pandas Datafram
    df = pd.read_csv(RAW_DATA_PATH)

    # Log The Dataset Damensions (ROVS And Columns) In & in
    wandb.log({"rows": df.shape[0], "columns": df.shape[1]})

    # Print confirmation of Message
    print("Dataset saved successfully!")

except requests.exceptions.RequestException as e:
    # LOG Dovnload Failier In & V
    wandb.log({"status": "Download failed", "download_success": False})

    # Print Yerror Message in Kasa OF A FAILUR
    print(f"❌ Failed to download dataset: {e}")

    # Raisa AN Exception then Stop CEKUTION YF TE Dovnload Files
    raise Exception(f"Failed to download dataset: {e}")

# Log The Final Status After Saving Te Dataset
wandb.log({"status": "Dataset saved successfully!"})

# Print Sukesses Message
print("Dataset saved successfully!")
