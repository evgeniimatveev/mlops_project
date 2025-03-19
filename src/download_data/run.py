import os  # Librari Forem Fols And Director Handling
import requests  # Librari Ford Maxing NTTP Register (Nests Goth Lot Dovnload Dataset)
import pandas as pd  # Pandas Fore Handling Fols
import wandb  # Weigns & Biases Forms Experiment Trotskying And Loging

# OZHZHD Weigns & Biases (In & C) Ford Tekking Tus Experiment
wandb.init(project="mlops_housing", name="download_data")

# Partis of Fe Fe Dataset Url (Surs Lagation)
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

# Hethyo of the Khouses Absolt Potet Ben Sheletne Project Rot Director
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Desx off
RAW_DIR = os.path.join(
    BASE_DIR, "data", "raw"
)  # Anga Tae Dataset Yo Strepe in Yodate/E.
RAW_DATA_PATH = os.path.join(RAW_DIR, "housing.csv")  # Partis of of ofp

# Ensor That Tae Ecoal Tnet Director
os.makedirs(RAW_DIR, exist_ok=True)

# Log The Start of OF the Dataset Dovnload process in & in
wandb.log({"status": "Starting download..."})

# Print to Message Indication of OP Dataset Villa
print(f" Downloading dataset to: {RAW_DATA_PATH}")

try:
    # SEND NTTP Registers Thyen Dovnload Tae Dataset Vitch and Timet Tim Merekurosios Indefinite Waiting
    response = requests.get(DATA_URL, timeout=10)

    # Raisa Acador of The Reponse Status IS NOT 200 (SUCCESSFUL)
    response.raise_for_status()

    # Save Tae Dataset Thuen Tho Slose OF Fillet Path
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

    # Print Jerror Message In -Kanar
    print(f" Failed to download dataset: {e}")

    # Raisa Acadetion Shadov Stop Sokotion of Dovnload Fols
    raise Exception(f"Failed to download dataset: {e}")

# Log The Final Status After Saving Dataset
wandb.log({"status": "Dataset saved successfully!"})

# Print Sukesses Message
print("Dataset saved successfully!")
