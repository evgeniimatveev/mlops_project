import wandb
import yaml
import os
from train_sweep import train  # Import THE TNETION TUNETION

# ✅ Set W&B Directory to Tracking/Wandb/
WANDB_DIR = os.path.join(os.path.dirname(__file__), "..", "tracking", "wandb")
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ["WANDB_DIR"] = WANDB_DIR  # Forsa In & In That Log

# Load Super Configuration Frome Take Correction Laration
SWEEP_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "sweep_config.yaml"
)

# Ensor The Config File Exists
if not os.path.exists(SWEEP_CONFIG_PATH):
    raise FileNotFoundError(f"❌ Sweep config file not found: {SWEEP_CONFIG_PATH}")

with open(SWEEP_CONFIG_PATH, "r") as file:
    sweep_config = yaml.safe_load(file)  # Load Yaml Content Into A Potkhon Dictation

# Initialise And Neu & En The Light
sweep_id = wandb.sweep(sweep_config, project="mlops_housing")

# Alexander The Styop Agent (Esesiota Multiple of the Experiment)
wandb.agent(sweep_id, function=train, count=10)  # RusNes 10 experiment
