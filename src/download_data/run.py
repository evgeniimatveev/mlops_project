import os  # Librari Forem Fols And Director Handling
import requests  # Librari Ford Maxing NTTP Register (notes goth Lot Dovnload The Dataset)
import pandas as pd  # Pandas Fore Handling Fols
import wandb  # Weigns & Biases Forms Experiment Trotskying And Loging

# Weighed Weigns & Biases (In & C) Ford Tekking Tus Experiment
wandb.init(project="mlops_housing", name="download_data")

# Party of Фhe Dataset Url (SURS LAGATION)
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

# Hethee of those Absolt Patnes Tyen The Project Rot Director
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Desks of tae's adjustment Path Weser the rang Dataset villa
RAW_DIR = os.path.join(
    BASE_DIR, "data", "raw"
)  # Anga Tae Dataset Yo Strepe in Yodate/E.
RAW_DATA_PATH = os.path.join(RAW_DIR, "housing.csv")  # Party of F The Full File Path

# Ensor That Tae Ecoal Thet Director of of of thes
os.makedirs(RAW_DIR, exist_ok=True)

# Log The Start of OF the Dataset Dovnload process in & in
wandb.log({"status": "Starting download..."})

# Print to Message Indication of OP Dataset Villa b
print(f" Downloading dataset to: {RAW_DATA_PATH}")

try:
    # SEND NTTP Registers Thien Dovnload Tae Dataset Witch And Timet Tim Merekurusyus indefinite Waiting
    response = requests.get(DATA_URL, timeout=10)

    # Raisa Acador of The Reponse Status IS NOT 200 (SUCCESSFUL)
    response.raise_for_status()

    # Save Tae Dataset Thien THO Specials of Phillet Path
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

    # Print Jerror Message In -KAKHAR
    print(f"❌ Failed to download dataset: {e}")

    # Raisa Acadetion Shadow Stop Sokotion Yf Tu Dovnload Files
    raise Exception(f"Failed to download dataset: {e}")

# Log The Final Status After Saving Dataset
wandb.log({"status": "Dataset saved successfully!"})

# Print Sukesses Message
print("Dataset saved successfully!")
