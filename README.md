MLOps Project 🚀

📌 Overview

This repository contains an end-to-end MLOps pipeline for managing, tracking, and automating machine learning experiments. The project integrates MLflow, Weights & Biases (W&B), SQL for analysis, and CI/CD automation.

📂 Project Structure 📁

MLOps_Project/
├── data/                  # Raw and processed datasets
├── mlruns/                # MLflow tracking logs
├── models/                # Saved models
├── notebook/              # Jupyter notebooks for analysis
├── sql_queries/           # SQL scripts for MLflow experiments analysis
├── src/                   # Core source code
│   ├── clean_data/        # Data preprocessing scripts
│   ├── download_data/     # Data downloading scripts
│   ├── feature_engineering/ # Feature transformation scripts
│   ├── model_training/    # Model training scripts
│   ├── model_deployment/  # API for model deployment
│   ├── utils/             # Utility functions
├── sweeps/                # W&B sweep scripts for hyperparameter tuning
├── tracking/              # Configurations for MLflow & W&B tracking
├── config/                # Project configuration files
├── .github/workflows/     # CI/CD pipeline
├── .gitignore             # Ignore unnecessary files
├── environment.yaml       # Conda environment dependencies
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── README_RU.md           # Russian documentation

🛠 Tech Stack

MLflow 🧪 – Experiment tracking and model registry

Weights & Biases (W&B) 📊 – Logging and hyperparameter sweeps

PostgreSQL 🛢️ – SQL for tracking and querying experiments

XGBoost 🌲 – Machine learning model

Python 🐍 – Main programming language

GitHub Actions ⚙️ – CI/CD automation (planned)

🔧 Setup & Installation

1️⃣ Clone the repository

git clone https://github.com/your-username/mlops_project.git
cd mlops_project

2️⃣ Create a virtual environment (Optional)

conda env create -f environment.yaml
conda activate mlops_env

OR

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
pip install -r requirements.txt

3️⃣ Run the pipeline

Run data preprocessing

python src/clean_data/run.py

Run model training

python src/model_training/run.py

Run hyperparameter tuning with W&B

python sweeps/sweep.py

Start MLflow UI

mlflow ui --host 0.0.0.0 --port 5000

Then open http://localhost:5000 in your browser.

🚀 Future Plans

✅ MLflow & W&B integration

✅ SQL experiment analysis

🔜 CI/CD with GitHub Actions

🔜 Model deployment via API

⚡ Happy Coding & Experiment Tracking! 🚀

