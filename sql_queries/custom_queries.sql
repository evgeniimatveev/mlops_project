-- custom_queries.sql
-- This script contains useful SQL queries for analyzing, monitoring, and reviewing MLflow experiments.

-- Query 1: Count total runs per experiment
SELECT 
    e.experiment_id, 
    e.name AS experiment_name, 
    COUNT(r.run_uuid) AS total_runs
FROM experiments e
LEFT JOIN runs r ON e.experiment_id = r.experiment_id
GROUP BY e.experiment_id, e.name
ORDER BY total_runs DESC;

-- Query 2: Retrieve the top 3 best runs based on RMSE (lower is better)
SELECT 
    r.run_uuid, 
    e.name AS experiment_name, 
    m.value AS rmse
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE m.key = 'RMSE'
ORDER BY m.value ASC
LIMIT 3;

-- Query 3: Get average RMSE, MAE, and R2 Score per experiment
SELECT 
    e.experiment_id, 
    e.name AS experiment_name,
    AVG(m.value) FILTER (WHERE m.key = 'RMSE') AS avg_rmse,
    AVG(m.value) FILTER (WHERE m.key = 'MAE') AS avg_mae,
    AVG(m.value) FILTER (WHERE m.key = 'R2 Score') AS avg_r2_score
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics m ON r.run_uuid = m.run_uuid
GROUP BY e.experiment_id, e.name
ORDER BY avg_rmse ASC; -- Sorting by best RMSE

-- Query 4: Analyze hyperparameter impact on RMSE
SELECT 
    p.key AS hyperparam,
    p.value AS param_value,
    m.value AS avg_rmse
FROM params p
JOIN metrics m ON p.run_uuid = m.run_uuid
WHERE m.key = 'RMSE'
ORDER BY avg_rmse ASC;
