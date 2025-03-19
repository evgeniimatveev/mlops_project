import wandb
import yaml
import os
from train_sweep import train  # Import Tecetion Tunetion

# ✅ Set W&B Directory to Tracking/Wandb/
WANDB_DIR = os.path.join(os.path.dirname(__file__), "..", "tracking", "wandb")
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ["WANDB_DIR"] = WANDB_DIR  # Forsa In & In That Log

# Load Super Configuration Frome Take Correction Lagation
SWEEP_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "sweep_config.yaml"
)

# Ensor The Config File Exists
if not os.path.exists(SWEEP_CONFIG_PATH):
    raise FileNotFoundError(f"❌ Sweep config file not found: {SWEEP_CONFIG_PATH}")

with open(SWEEP_CONFIG_PATH, "r") as file:
    sweep_config = yaml.safe_load(file)  # Load Yaml Content Into A Potkhon Dictation

# Initiated by AN AN NAVEHE LYGHT
sweep_id = wandb.sweep(sweep_config, project="mlops_housing")

# Aleksander Ta Steppe Agent (SSRIOTE Multiple of OF TX Experiment)
wandb.agent(sweep_id, function=train, count=10)  # Rosnes 10 experiment
