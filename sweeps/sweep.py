import wandb
import yaml
import os
from train_sweep import train  # Import the training function

# ✅ Set W&B directory to tracking/wandb/
WANDB_DIR = os.path.join(os.path.dirname(__file__), "..", "tracking", "wandb")
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ["WANDB_DIR"] = WANDB_DIR  # Force W&B to log here

#  Load sweep configuration from the correct location
SWEEP_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "sweep_config.yaml"
)

#  Ensure the config file exists
if not os.path.exists(SWEEP_CONFIG_PATH):
    raise FileNotFoundError(f"❌ Sweep config file not found: {SWEEP_CONFIG_PATH}")

with open(SWEEP_CONFIG_PATH, "r") as file:
    sweep_config = yaml.safe_load(file)  # Load YAML content into a Python dictionary

#  Initialize a new W&B Sweep
sweep_id = wandb.sweep(sweep_config, project="mlops_housing")

# ‍♂️ Run the sweep agent (executes multiple experiments)
wandb.agent(sweep_id, function=train, count=10)  # Run 10 experiments
