# MLOps Pipeline — XGBoost · MLflow · W&B · PostgreSQL · GitHub Actions

![MLOps](https://img.shields.io/badge/MLOps-Automation-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)
![W&B](https://img.shields.io/badge/W%26B-Sweeps-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Experiments-blue?logo=postgresql)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-black?logo=githubactions)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## What This Project Does

End-to-end MLOps pipeline for training, tracking, and analyzing XGBoost experiments. Dual tracking with **MLflow** (local registry) and **Weights & Biases** (cloud logging + hyperparameter sweeps), with **PostgreSQL** used to query the MLflow database directly via SQL.

**What makes it stand out:** SQL analytics on top of the MLflow experiment database — query best runs, compare hyperparameters, and analyze experiment history the same way you'd query business data.

**Pipeline:** `Data → Preprocessing → Feature Engineering → XGBoost → MLflow + W&B → SQL Analysis → CI/CD`

---
  **Best run results (W&B Sweep — 10 Bayesian runs):**
  | Metric | Value |
  |--------|-------|
  | Best R² Score | **0.8326** |
  | Best RMSE | **$46,833** |
  | Best MAE | **$30,185** |
  | GridSearch evaluations | 729 (243 configs × 3-fold CV) |
  | W&B Sweep runs | 10 (Bayesian optimization, 5 hyperparameters) |

  **Dataset:** California Housing — 20,640 records, predicting median house value

---
## SQL on MLflow Experiments

Most people use the MLflow UI. This project queries the MLflow PostgreSQL backend directly with SQL.

### Best 3 Runs by RMSE

```sql
SELECT
    r.run_uuid,
    e.name          AS experiment_name,
    m.value         AS rmse
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics     m ON r.run_uuid      = m.run_uuid
WHERE m.key = 'RMSE'
ORDER BY m.value ASC
LIMIT 3;
```

### Hyperparameter Impact on RMSE

```sql
SELECT
    p.key            AS hyperparam,
    p.value          AS param_value,
    AVG(m.value)     AS avg_rmse
FROM runs    r
JOIN params  p ON r.run_uuid = p.run_uuid
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE m.key = 'RMSE'
GROUP BY p.key, p.value
ORDER BY avg_rmse ASC;
```

### Experiment Summary (Runs Count)

```sql
SELECT
    e.experiment_id,
    e.name           AS experiment_name,
    COUNT(r.run_uuid) AS total_runs
FROM experiments e
LEFT JOIN runs r ON e.experiment_id = r.experiment_id
GROUP BY e.experiment_id, e.name
ORDER BY total_runs DESC;
```

---

## Architecture

```
Raw Data
    └── src/clean_data/          # preprocessing
            └── src/feature_engineering/
                    └── src/model_training/   # XGBoost
                            ├── MLflow        # local experiment registry
                            ├── W&B           # cloud logging + sweeps
                            └── PostgreSQL    # SQL on MLflow DB
                                    └── sql_queries/
                                            ├── best_models.sql
                                            ├── hyperparams_analysis.sql
                                            ├── experiments_analysis.sql
                                            └── custom_queries.sql
```

---

## Project Structure

```
mlops_project/
├── src/
│   ├── clean_data/           # data preprocessing
│   ├── download_data/        # data ingestion
│   ├── feature_engineering/  # feature transformations
│   ├── model_training/       # XGBoost training + logging
│   ├── model_deployment/     # FastAPI inference endpoint
│   └── utils/                # shared utilities
├── sql_queries/              # SQL analytics on MLflow DB
│   ├── best_models.sql
│   ├── hyperparams_analysis.sql
│   ├── experiments_analysis.sql
│   ├── custom_queries.sql
│   └── delete_old_experiments.sql
├── sweeps/                   # W&B hyperparameter sweep configs
├── config/                   # project configuration
├── models/                   # saved model artifacts
├── .github/workflows/        # CI/CD pipeline
└── environment.yaml
```

---

## How to Run

```bash
# 1. Clone and set up environment
git clone https://github.com/evgeniimatveev/mlops_project.git
cd mlops_project
conda env create -f environment.yaml
conda activate mlops_env

# 2. Run pipeline steps
python src/clean_data/run.py
python src/model_training/run.py

# 3. Launch MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Open http://localhost:5000

# 4. Run W&B hyperparameter sweep
python sweeps/sweep.py

# 5. Query experiments via SQL
# Connect DBeaver to PostgreSQL → run sql_queries/*.sql
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Model | XGBoost |
| Experiment Tracking | MLflow (local) + W&B (cloud) |
| Hyperparameter Tuning | W&B Sweeps |
| Experiment Analytics | PostgreSQL + SQL |
| CI/CD | GitHub Actions |
| Language | Python |

---

## Connect

- GitHub: [evgeniimatveev](https://github.com/evgeniimatveev)
- Portfolio: [datascienceportfol.io/evgeniimatveevusa](https://www.datascienceportfol.io/evgeniimatveevusa)
- LinkedIn: [Evgenii Matveev](https://www.linkedin.com/in/evgenii-matveev-510926276/)
